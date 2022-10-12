import os
import time
import json
import numpy as np
import torch
from .utils import load_weights, to_numpy, to_tensor
from .foldingnet import FoldingNet
from .pointclouds import fragment_pointcloud_exact, reassemble_pointcloud, optimal_assignments_mcf
from .density import apply_design_pc


def load_network(weights: str, emd: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert os.path.isfile(weights)
    factors = {"EMD": 0.18, "chamfer": 0.03}
    f = os.path.join(os.path.dirname(weights), "factors.json")
    if os.path.isfile(f):
        factors = json.load(open(f, "r"))
    extra_kwargs = {"batch_size": 8, "chamfer_factor": 1.0, "emd": emd, "emdscaling": 0.8, "loss_factors": factors}

    net = FoldingNet(num_points=2048, shape="cube", feat_dim=512, k=16, **extra_kwargs)
    net.to(device)
    w = load_weights(weights, device)
    net.load_state_dict(w)
    return net


def apply_net(model, x: np.ndarray, n: int):
    x = to_numpy(x)
    pc, centroids = fragment_pointcloud_exact(x, n)
    device = next(iter(model.parameters())).device
    z = to_tensor(np.concatenate([np.expand_dims(y, 0) for y in pc], axis=0), device).float()
    with torch.no_grad():
        out = model(z)
    return to_numpy(reassemble_pointcloud(out, to_tensor(centroids, device)))


def interpolate_pcs(model, pc_clustered, centroids, pc2, factor: float = 0.5, verbose: bool = True):
    if isinstance(pc_clustered, list):
        pc_clustered = np.concatenate([np.expand_dims(to_numpy(y), 0) for y in pc_clustered], axis=0)
    else:
        pc_clustered = to_numpy(pc_clustered)
    if isinstance(centroids, list):
        centroids = np.concatenate([to_numpy(y) for y in centroids])
    else:
        centroids = to_numpy(centroids)

    pc_clustered2, centroids2 = optimal_assignments_mcf(centroids, pc2, verbose=verbose)
    centroids2 = np.concatenate([np.expand_dims(y, 0) for y in centroids2])
    device = next(iter(model.parameters())).device
    C = to_tensor((1 - factor) * centroids + factor * centroids2, device).float()
    x1 = to_tensor(pc_clustered, device)
    x2 = to_tensor(np.concatenate([np.expand_dims(y, 0) for y in pc_clustered2], axis=0).astype("float32"), device)
    assert x1.size(0) == x2.size(0)
    features1 = model.encoder(x1)
    features2 = model.encoder(x2)
    out = model.decoder((1 - factor) * features1 + factor * features2)
    return to_numpy(reassemble_pointcloud(out, C)).squeeze()


def apply_design(model, x: np.ndarray, design: np.ndarray, n: int, factor: float = 0.5, verbose: bool = True):
    """
    Apply design from design point cloud :code:`design` to the pointcloud :code:`x`
    :code:`factor` should be in [0,1] and the higher the more the design is applied.
    """
    x = to_numpy(x)
    t0 = time.time()
    pc, centroids = fragment_pointcloud_exact(x, n)
    if verbose:
        print("[Info] Clustered point cloud in {}s".format(round(time.time() - t0, ndigits=2)))
    t0 = time.time()
    z = apply_design_pc(x, design, n_jobs=8)
    if verbose:
        print("[Info] Applied design in {}s".format(round(time.time() - t0, ndigits=2)))
    t0 = time.time()
    out = interpolate_pcs(model, pc, centroids, z, factor, verbose)
    if verbose:
        print("[Info] Applied network in {}s".format(round(time.time() - t0, ndigits=2)))
    return out
