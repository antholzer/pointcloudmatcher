import os
import math
import torch
import random
import json
import numpy as np
from os.path import join
from glob import glob
from typing import List, Union
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from .utils import load_blz, to_numpy


class _FolderData(torch.utils.data.DataLoader):
    def __init__(
        self,
        fend: Union[List[str], str],
        paths: Union[List[str], str],
        split: float = 0.8,
        train: bool = True,
        transform=None
    ):
        if not isinstance(paths, list):
            paths = [paths]
        self.paths, self.transform = paths, transform
        if not all(os.path.isdir(path) for path in self.paths):
            raise ValueError(f"{paths} is not a folder")
        if not isinstance(fend, list):
            fend = [fend]
        filelist: List[str] = []
        for path in self.paths:
            for ff in fend:
                filelist += glob(join(path, f"*.{ff}")) + glob(join(path, f"*/*.{ff}")
                                                              ) + glob(join(path, f"*/*/*.{ff}"))

        # make sure to not include folders that have fend in their name
        filelist = list(filter(lambda x: os.path.isfile(x), filelist))
        random.seed(123)
        random.shuffle(filelist)
        n = int(len(filelist) * split)
        if train:
            self.filelist = filelist[:n]
        else:
            self.filelist = filelist[n:]

    def __len__(self) -> int:
        return len(self.filelist)

    def save_filelist(self, out):
        with open(out, "w") as f:
            for x in self.filelist:
                f.write(x)
                f.write("\n")


class xyzData(_FolderData):
    def __init__(
        self,
        path: Union[List[str], str],
        num_points: int = None,
        split: float = 0.8,
        train: bool = True,
        block: bool = True,
        ord: int = 2,
        transform=None
    ):
        super().__init__("xyz", path, split, train, transform)
        self.num_points = num_points
        self.ord, self.block = ord, block

    def __getitem__(self, idx: int) -> torch.Tensor:
        pc = np.loadtxt(self.filelist[idx]).astype("float32")
        if self.num_points is not None and pc.shape[0] > self.num_points:
            if self.block:
                pc = random_block(pc, self.num_points, self.ord)
            pc = pc[:self.num_points]
        x = torch.from_numpy(pc)
        if self.transform is not None:
            return self.transform(x)
        else:
            return x


class BloscData(_FolderData):
    def __init__(
        self,
        path: Union[List[str], str],
        num_points: int = None,
        split: float = 0.8,
        train: bool = True,
        block: bool = True,
        ord: int = 2,
        transform=None
    ):
        super().__init__(["blosc", "blz"], path, split, train, transform)
        self.num_points = num_points
        self.ord, self.block = ord, block

    def __getitem__(self, idx: int) -> torch.Tensor:
        pc = load_blz(self.filelist[idx]).astype("float32").reshape(-1, 3).copy()
        if self.num_points is not None and pc.shape[0] > self.num_points:
            if self.block:
                pc = random_block(pc, self.num_points, self.ord)
            pc = pc[:self.num_points]
        x = torch.from_numpy(pc)
        if self.transform is not None:
            return self.transform(x)
        else:
            return x


class RandomRotation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = rotation_matrix(*np.random.rand(3) * 2 * np.pi).to(x.device)
        return x @ R


class Normalize(torch.nn.Module):
    """
    normalizes pc the same way as our Shapenet dataset

    box_center=True -> our normalization, better centering for base pointclounds when applying design
    """
    def __init__(self, box_center=False):
        super().__init__()
        self.r = torch.tensor(1.0)
        self.M = torch.tensor(0.0)
        self.box_center = box_center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.box_center:
            if x.ndim == 3:
                m1, m2 = x.min(dim=(0, 1)).values, x.max(dim=(0, 1)).values
            else:
                m1, m2 = x.min(dim=0).values, x.max(dim=0).values
            M = (m1 + m2) / 2
        else:
            if x.ndim == 3:
                M = x.mean(dim=(1, 2), keepdim=True)
            else:
                M = x.mean()
        x = x - M
        r = (x[..., 0]**2 + x[..., 1]**2 + x[..., 2]**2).max()
        r = torch.sqrt(r)
        self.r, self.M = r, M
        return x / r / 2

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.M) / self.r / 2

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2 * self.r
        return x + self.M

    def to_numpy(self) -> List[np.ndarray]:
        return [to_numpy(self.r), to_numpy(self.M)]


def denormalize(x, vals):
    x = x * 2 * vals[0]
    return x + vals[1]


def rotation_matrix(a, b, c):
    rz = torch.tensor([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=torch.float32)
    ry = torch.tensor([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]], dtype=torch.float32)
    rx = torch.tensor([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]], dtype=torch.float32)
    return rz @ (ry @ rx)


def random_block(x: np.ndarray, n: int, p: float = 2) -> np.ndarray:
    index = np.random.randint(x.shape[0], size=1)
    refpoint = x[index, :]
    distances = np.zeros(x.shape[0], dtype=x.dtype)
    for j in range(x.shape[0]):
        distances[j] = np.linalg.norm(distances[j] - refpoint, ord=p)
    indices = np.argsort(distances)[:n]
    points = x[indices, :]
    return points - points.mean(axis=0)


def max_pw_distance(d, metric, device):
    m = 0
    for i in tqdm(range(len(d))):
        for j in range(i + 1, len(d)):
            dist = metric(torch.unsqueeze(d[i], 0).to(device), torch.unsqueeze(d[j], 0).to(device))
            m = max(m, dist)
    if torch.is_tensor(m):
        m = m.item()
    return m


def max_emd_chamfer_distances(file: str, dset, portion: float = 0.1, comment: str = ""):
    from .distances import EMDLoss

    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            d = json.loads(text)
    else:
        p = min(100, 100 * portion)
        print(f"Calculating distances for {p}% of the dataset")
        if portion < 1:
            dset = torch.utils.data.Subset(dset, torch.arange(math.floor(len(dset) * portion)))

        d = {}
        d["EMD"] = max_pw_distance(dset, EMDLoss(), torch.device('cuda'))
        d["chamfer"] = max_pw_distance(dset, lambda x, y: chamfer_distance(x, y)[0].item(), torch.device('cuda'))
        if len(comment) > 1:
            d["comment"] = comment
        d["split"] = portion
        d["dataset"] = str(dset)
        with open(file, "w") as f:
            f.write(json.dumps(d, indent=2, sort_keys=True) + "\n")
    return d
