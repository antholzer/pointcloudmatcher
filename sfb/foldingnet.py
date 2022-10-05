import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch3d.ops.knn import knn_points, knn_gather
from pytorch3d.loss import chamfer_distance
from typing import Union, Tuple
from functools import reduce
from operator import mul
from .utils import get_lr_scheduler


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    B = x.size(0)
    # x : [batch, N, 3]
    inner = x @ x.transpose(2, 1)
    norm2 = torch.cat([torch.diag(inner[k]).unsqueeze_(0) for k in range(B)]).unsqueeze_(-1)
    pairwise_distances = norm2 - 2 * inner + norm2.transpose(1, 2)
    idx = pairwise_distances.topk(k=k, dim=1, largest=False).indices
    idx = idx.transpose(1, 2).reshape(B, -1, 1).contiguous()  # [B, k*N, 1]
    return idx


def local_covariance(pc: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # pc [B, N, 3]
    B, d = pc.size(0), pc.size(2)
    num_points = pc.size(1)  # N
    x = knn_gather(pc, idx)
    # sample covariance
    x = x - torch.mean(x, dim=2, keepdim=True)
    x = (x.transpose(2, 3) @ x).view(B, num_points, d * d)
    return torch.cat((pc, x), dim=2)


def local_maxpool(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)  # [B, N, d]
    x = knn_gather(x, idx)
    x, _ = torch.max(x, dim=2)
    return x


class FoldingNet_Encoder(nn.Module):
    def __init__(self, feat_dim=512, k=16):
        super().__init__()
        self.k = k
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, feat_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim, feat_dim, 1),
        )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        _, idx, _ = knn_points(pts, pts, K=self.k, return_sorted=False)
        x = local_covariance(pts, idx).transpose(1, 2)  # (B, 12, num_points])
        x = self.mlp1(x)
        x = self.graph_layer(x, idx)  # (B, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]
        x = self.mlp2(x)
        feat = x.transpose(2, 1)  # (B, 1, feat_dim)
        return feat


def _find_m(n: int, k: int = 2):
    m = int(n**(1 / k))
    M = [m for _ in range(k)]
    i = -1
    while reduce(mul, M) < n:
        i += 1
        M[i % k] += 1
    M[i % k] -= 1
    return M


class FoldingNet_Decoder(nn.Module):
    def __init__(self, shape="plane", feat_dim=512, m=45, intermediate=False, num_points=None, search_m=True):
        super().__init__()
        self.shape = shape
        if self.shape == "plane":
            grid_dim = 2
            self.m = m**2
        elif self.shape == "cube":
            grid_dim = 3
            self.m = m**3
        else:
            self.m = m
            grid_dim = 0

        if num_points is not None and search_m:
            self.grid_dims = _find_m(num_points, grid_dim)
            self.m = reduce(mul, self.grid_dims)
        else:
            self.grid_dims = [m for _ in range(grid_dim)]
        if grid_dim > 0:
            self.meshgrid = [[-0.3, 0.3, k] for k in self.grid_dims]

        self.folding1 = nn.Sequential(
            nn.Conv1d(feat_dim + grid_dim, feat_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim, feat_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(feat_dim + 3, feat_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim, feat_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim, 3, 1),
        )
        self.grid = None
        self.intermediate = intermediate

    def build_grid(self, batch_size, device=torch.device("cpu")):
        if self.shape is None:
            return None
        elif self.shape == 'plane':
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == "cube":
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            z = np.linspace(*self.meshgrid[2])
            points = np.array(list(itertools.product(x, y, z)))

        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points, dtype=torch.float32)
        return points.transpose(1, 2).to(device)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.transpose(1, 2).repeat(1, 1, self.m)  # (B, feat_dim, num_points)
        if self.grid is None and self.shape is not None:
            self.grid = self.build_grid(x.shape[0], x.device)
        if self.shape is not None:
            cat1 = torch.cat((x, self.grid), dim=1)
        else:
            cat1 = x
        folding_result1 = self.folding1(cat1)  # (B, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)
        folding_result2 = self.folding2(cat2)
        if self.intermediate:
            return folding_result2.transpose(1, 2).contiguous(), folding_result1.transpose(1, 2).contiguous()
        else:
            return folding_result2.transpose(1, 2).contiguous()  # (B, num_points ,3)


class FoldingNet(pl.LightningModule):
    """
    Can configure *feat_dim* (512, dim of feature space), *shape* ('plane' or None, the object that is 'folded'),
    *m* (45, the square root of the number of points in the output), *k* (16, parameters for the kNN layers)
    """
    def __init__(self, feat_dim=512, shape="plane", m=45, k=16, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if kwargs.get("num_points", None) is not None:
            if not kwargs.get("search_m", True):
                if shape == "cube":
                    m = int((kwargs["num_points"])**(1 / 3))
                else:
                    m = int(np.sqrt(kwargs["num_points"]))
            self.num_points = kwargs["num_points"]
        else:
            self.num_points = m**3 if shape == "cube" else m**2

        if kwargs.get("batch_size", None) is not None:
            self.example_input_array = torch.empty(kwargs["batch_size"], self.num_points, 3)

        self.config = {"name": "FoldingNet", "feat_dim": feat_dim, "shape": shape, "m": m, "k": k, **kwargs}
        self.eps = kwargs.get("epsilon", kwargs.get("eps", 0))
        self.emd = kwargs.get("emd", kwargs.get("earthmover", kwargs.get("wasserstein", False)))
        self.chamfer_factor = kwargs.get("chamfer_factor", kwargs.get("chamfer", 0.0))
        self.feat_dim = feat_dim
        self.encoder = FoldingNet_Encoder(feat_dim, k)
        self.decoder = FoldingNet_Decoder(
            shape, feat_dim, m, kwargs.get("intermediate", False), kwargs.get("num_points", None),
            kwargs.get("search_m", True)
        )
        self.factors = {}
        if kwargs.get("loss_factors", None) is not None:
            self.factors["chamfer"] = 1.0 / kwargs["loss_factors"]["chamfer"]
            self.factors["emd"] = 1.0 / kwargs["loss_factors"]["EMD"]
        elif self.emd:
            self.factors["emd"], self.factors["chamfer"] = 1, 1
            print("EMD/Chamfer weights not given. Might yield suboptimal results.")
        if self.emd:
            from .distances import EMDLoss
            self.emd_loss = EMDLoss(kwargs.get("p", 2), scaling=kwargs.get("emdscaling", 0.9))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(input)
        return self.decoder(feature)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_epoch=True)
        # return loss

    def shared_step(self, batch):
        x = batch
        if self.eps > 0:
            out = self(x + torch.randn_like(x) * self.eps)
        else:
            out = self(x)
        if self.emd:
            loss = self.emd_loss(out, x)
            if self.chamfer_factor > 0:
                ch_loss, _ = chamfer_distance(out, x)
                loss = loss * self.factors["emd"] + self.chamfer_factor * ch_loss * self.factors["chamfer"]
        else:
            loss, _ = chamfer_distance(out, x)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.config.get("lr", 0.001), weight_decay=self.config.get("weight_decay", 0)
        )
        sched = get_lr_scheduler(opt, self.config)
        if sched is not None:
            if isinstance(sched, dict):
                sched["monitor"] = "val_loss"
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt

    def reset(self):
        self.decoder.grid = None
