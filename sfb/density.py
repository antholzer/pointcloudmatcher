import os
import torch
import numpy as np
from sklearn.neighbors import KernelDensity
from joblib import delayed, Parallel
from tqdm import tqdm
from typing import List


def apply_design_pc(pc, design_pc, bandwidth=0.01, replace=True, jitter=True, n_jobs=1):
    if torch.is_tensor(pc):
        pc = pc.detach().cpu().numpy()
    if torch.is_tensor(design_pc):
        design_pc = design_pc.detach().cpu().numpy()
    if pc.ndim > 2:
        pc = pc.squeeze()
    if design_pc.ndim > 2:
        design_pc = design_pc.squeeze()
    assert (pc.ndim == 2 and design_pc.ndim == 2)
    pc = pc.copy()

    kde = KernelDensity(bandwidth=bandwidth).fit(pc)
    if n_jobs > 1:
        density = Parallel(n_jobs=n_jobs)(delayed(kde.score_samples)(design_pc[i:i + 1]) for i in range(len(design_pc)))
        density = np.exp(np.concatenate(density).reshape(-1))
    else:
        density = np.exp(kde.score_samples(design_pc))
    density = density / density.sum()
    n = pc.shape[0]
    i = np.random.choice(range(len(density)), n, replace=replace, p=density)
    if jitter and replace:
        i.sort()
        duplicates = i[1:][(i[:-1] - i[1:]) == 0]
        if len(duplicates) == 0:
            return design_pc[i]
        randsigns = (1 + 2 * np.random.randint(-1, 1, (len(duplicates), 3)))
        npoints = design_pc[duplicates]
        npoints = npoints + randsigns * (0.001 * npoints)
        return np.concatenate((design_pc[np.unique(i)], npoints), axis=0)
    else:
        return design_pc[i]


def normalize_design_pc(pc: np.ndarray) -> np.ndarray:
    pc = pc - pc.mean()
    pc = pc / max(pc[:, k].max() for k in range(3))
    return pc / 2


def apply_designs_pcs(pcs, design_pcs, bandwidth=0.01, replace=True, n_jobs=1, use_tqdm=True):
    if not isinstance(design_pcs, List):
        design_pcs = [design_pcs for _ in range(len(pcs))]
    N = min(len(pcs), len(design_pcs))

    def f(x, y):
        o = apply_design_pc(x, y, bandwidth=bandwidth, replace=replace)
        return o

    if n_jobs > 1:
        out = ProgressParallel(total=N, use_tqdm=use_tqdm, max_nbytes=None,
                               n_jobs=n_jobs)(delayed(f)(pcs[i], design_pcs[i]) for i in range(N))
    else:
        out = [f(pcs[i], design_pcs[i]) for i in tqdm(range(N))]
    return out


def apply_designs_pcs_to_dataset(data, design_pcs, out_dir, n_jobs=1):
    """
    Example:

        data = sfb.data.xyzData(["/data/210407_Training_basic/", "/data/210822_combined/"], split=1,
                                       transform=sfb.data.Normalize(box_center=True))
        design_pc_files = [...]
        design_pcs = {os.path.basename(f): normalize_design_pc(np.loadtxt(f)) for f in design_pc_files}
        apply_designs_pcs_to_dataset(data, design_pcs, "/data/out", n_jobs=8)
    """
    loader = torch.utils.data.DataLoader(data, num_workers=4, drop_last=False, batch_size=n_jobs * 2)
    k = 0
    if not isinstance(design_pcs, dict):
        design_pcs = {"design": design_pcs}

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for key in design_pcs.keys():
        odir = os.path.join(out_dir, key)
        os.mkdir(odir)
        design_pc = design_pcs[key]

        for b in tqdm(loader):
            out = apply_designs_pcs(b, design_pc, replace=True, n_jobs=n_jobs, use_tqdm=False)
            for i in range(len(out)):
                np.savetxt(os.path.join(odir, f"{k}.xyz"), out[i])
                k += 1


# https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
