import os
import time
import numpy as np
import scipy.cluster.vq
import torch
from tqdm import trange
from kmeans_pytorch import kmeans
from k_means_constrained import KMeansConstrained
from k_means_constrained.k_means_constrained_ import minimum_cost_flow_problem_graph, solve_min_cost_flow_graph
from .utils import to_numpy, to_tensors


def fragment_pointcloud(x: np.ndarray, n: int, p: float = 2):
    points = x.copy()
    if len(x.shape) > 2:
        x = x.squeeze()
    k = x.shape[0] // n
    centroids, _ = scipy.cluster.vq.kmeans(x, k)
    if centroids.shape[0] < k:
        n_temp = k - centroids.shape[0]
        tmp = x[np.random.randint(0, x.shape[0], n_temp), :]
        centroids = np.concatenate((centroids, tmp), axis=0)
    distances = np.zeros(x.shape[0], dtype=x.dtype)
    pointclouds = []
    for k in range(centroids.shape[0]):
        distances = np.linalg.norm(points - centroids[k], ord=p, axis=1)
        indices = np.argsort(distances[:points.shape[0]])[:n]
        pointclouds.append(points[indices, :] - centroids[k])
        points = np.delete(points, indices, axis=0)
    else:
        return pointclouds, centroids


def fragment_pointcloud_exact(x: np.ndarray, n: int, **kwargs):
    """
    if :code:`cut=True` (not default) we throw away :code:`x.shape[0] % n` points.
    """
    if kwargs.get("cut", False):
        n1 = n
        n2 = x.shape[0] // (x.shape[0] // n1)
    else:
        assert x.shape[0] % n == 0
        n1, n2 = n, n
    k = x.shape[0] // n1
    n_jobs = kwargs.get("n_jobs", 8)
    K = KMeansConstrained(n_clusters=k, size_min=n1, size_max=n2, n_jobs=n_jobs)
    K.fit(x)
    centroids = K.cluster_centers_
    pointclouds = [x[K.labels_ == i] - centroids[i] for i in range(k)]
    if kwargs.get("cut", False):
        pointclouds = [p[:n1] for p in pointclouds]
    return pointclouds, centroids


def fragment_pointcloud_torch(x, n: int, p: float = 2):
    points = x.detach().clone()
    if len(x.size()) > 2:
        x = x.squeeze()
    k = x.shape[0] // n
    _, centroids = kmeans(X=x, num_clusters=k, distance='euclidean', device=x.device, tqdm_flag=False, iter_limit=1000)
    centroids = centroids.to(x.device)
    distances = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    pointclouds = []
    for k in range(centroids.shape[0]):
        distances = torch.linalg.norm(points - centroids[k], ord=p, dim=1)
        indices = torch.argsort(distances[:points.shape[0]])[:n]
        pointclouds.append(points[indices, :] - centroids[k])
        mask = torch.ones(points.size(), dtype=torch.bool, device=x.device)
        mask[indices, :] = False
        points = points[mask].view(points.size(0) - len(indices), 3)
    else:
        return pointclouds, centroids


def fragment_pointcloud_hierarchical_torch(x, n: int, p: float = 2, postprocess: bool = True, iter_limit=1000):
    points = x.detach().clone()
    if len(x.size()) > 2:
        points = points.squeeze()
    clusters = {0: points}
    centers = {}
    k = 1
    first_split = 2.1
    second_split = 1.6
    while not all(value.shape[0] <= first_split * n for value in clusters.values()):
        keys = list(clusters.keys())
        for key in keys:
            pc = clusters[key]
            if pc.shape[0] > first_split * n:
                idx, centroids = kmeans(
                    X=pc,
                    num_clusters=2,
                    distance='euclidean',
                    device=pc.device,
                    tqdm_flag=False,
                    iter_limit=iter_limit
                )
                pc1 = pc[idx == 0]
                clusters[k] = pc1
                centers[k] = centroids[0]
                k += 1
                pc2 = pc[idx == 1]
                clusters[k] = pc2
                centers[k] = centroids[1]
                k += 1
                clusters.pop(key)

    while not all(value.shape[0] <= second_split * n for value in clusters.values()):
        keys = list(clusters.keys())
        for key in keys:
            pc = clusters[key]
            if pc.shape[0] > second_split * n:
                idx, centroids = kmeans(
                    X=pc,
                    num_clusters=2,
                    distance='euclidean',
                    device=pc.device,
                    tqdm_flag=False,
                    iter_limit=iter_limit
                )
                pc1 = pc[idx == 0]
                clusters[k] = pc1
                centers[k] = centroids[0]
                k += 1
                pc2 = pc[idx == 1]
                clusters[k] = pc2
                centers[k] = centroids[1]
                k += 1
                clusters.pop(key)

    m = points.shape[0] // n
    if len(clusters) < m:
        diff = m - len(clusters)
        for j in range(diff):
            max_cluster = max(clusters, key=lambda x: clusters[x].shape[0])
            pc = clusters[max_cluster]
            idx, centroids = kmeans(
                X=clusters[max_cluster],
                num_clusters=2,
                distance='euclidean',
                device=pc.device,
                tqdm_flag=False,
                iter_limit=iter_limit
            )
            pc1 = pc[idx == 0]
            clusters[k] = pc1
            centers[k] = centroids[0]
            k += 1
            pc2 = pc[idx == 1]
            clusters[k] = pc2
            centers[k] = centroids[1]
            k += 1
            clusters.pop(max_cluster)

    if postprocess:
        for key in clusters:
            pc = clusters[key]
            if pc.shape[0] > n:
                indices = torch.randint(0, pc.shape[0], (n,))
                clusters[key] = pc[indices]
            if pc.shape[0] < n:
                up = simple_upsample(pc, n, 0.01)
                clusters[key] = up
    keys = list(sorted(clusters.keys()))
    return [clusters[k] - centers[k].to(x.device) for k in keys], [centers[k].to(x.device) for k in keys]


def fragment_dataset(
    dset, n: int, save_func, logdir: str, fname: str, hierarchical: bool = False, mcf: bool = True, **kwargs
):
    """
    Fragment dataset using :code:`fragment_pointcloud_exact` by default set :code:`mcf=False, hierarchical=True`
    in order to use hierarchical clustering (faster).

    Example

    d = sfb.data.xyzData("/data/210402_Training_basic", transform=sfb.data.Normalize(), split=1)
    fragment_dataset(d, 1024, sfb.utils.save_blz, "/data/1024Training_basic", "{}.blosc")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = 0
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    for i in trange(len(dset)):
        try:
            if mcf:
                x = dset[i].cpu().numpy().squeeze()
                pcs, _ = fragment_pointcloud_exact(x, n, **kwargs)
            elif hierarchical:
                x = dset[i].to(device)
                pcs, _ = fragment_pointcloud_hierarchical_torch(x, n, **kwargs)
                pcs = [pc.detach().cpu().numpy() for pc in pcs]
            else:
                pcs, _ = fragment_pointcloud(dset[i].cpu().numpy().squeeze().copy(), n, **kwargs)
        except ValueError:
            print(dset.filelist[i])
            continue
        for x in pcs:
            save_func(os.path.join(logdir, fname.format(k)), x)
            k += 1


def reassemble_pointcloud(x, centroids):
    if not len(x) == len(centroids):
        d1, d2 = len(x), len(centroids)
        raise ValueError("x has {} samples, centroids {}".format(d1, d2))
    for k in range(len(x)):
        x[k] += centroids[k]
    if torch.is_tensor(x[0]):
        return torch.cat([y for y in x])
    else:
        return np.concatenate([y for y in x])


def simple_upsample(pc: torch.Tensor, N: int, factor: float = 0.01) -> torch.Tensor:
    sq = False
    if pc.ndim < 3:
        pc = pc.unsqueeze(0)
        sq = True
    if pc.ndim != 3 or pc.size(-1) != 3:
        raise ValueError("Pointclouds has wrong dimension. Reshape it beforehand.")

    n = pc.size(1)
    if n >= N:
        if sq:
            return pc.squeeze(0)
        else:
            return pc[:, :N, :]

    d = N - n
    i = torch.randint(0, n, (pc.size(0), d), device=pc.device)
    ii = i.unsqueeze(-1).expand([i.size(0), i.size(1), 3])
    npoints = torch.gather(pc, 1, ii)
    randsigns = (1 + 2 * torch.randint(-1, 1, npoints.size(), device=npoints.device))
    npoints = npoints + randsigns * (factor * npoints)
    out = torch.cat((pc, npoints), dim=1)
    if sq:
        return out.squeeze(0)
    else:
        return out


def optimal_assignments_mcf(centroids: np.ndarray, pc: np.ndarray, verbose: bool = True):
    pc = to_numpy(pc)
    if isinstance(centroids, list):
        centroids = [to_numpy(c)[None, :] for c in centroids]
        centroids = np.concatenate(centroids)
    else:
        centroids = to_numpy(centroids)

    t0 = time.time()
    d, N = len(centroids), len(pc)
    n = N // d
    assert (N % d) == 0
    C = np.zeros((d, N), dtype=np.float32)
    for i in range(d):
        C[i] = np.sum((centroids[i] - pc)**2, axis=1)
    edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(pc, centroids, C.T, n, n)
    i = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

    if verbose:
        print("[Info] Solved MCF problem in {}s".format(round(time.time() - t0, ndigits=2)))
    x = [pc[i == k] for k in range(d)]
    pcs = torch.cat([xx.unsqueeze(0) for xx in to_tensors(x)])
    centroids = [pcs[i].mean(0) for i in range(pcs.size(0))]
    for i in range(pcs.size(0)):
        pcs[i] = pcs[i] - centroids[i]
    return pcs, centroids
