import os
import torch
import shutil
import time
import datetime
import subprocess
import blosc
import json
import random
import numpy as np
import open3d as o3d
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from open3d.visualization.tensorboard_plugin import summary  # noqa
from collections import OrderedDict
from typing import Union, List, Any


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def deterministic(b: bool = True):
    """always use the same (deterministic) algorithm for e.g. convolutions"""
    torch.use_deterministic_algorithms(b)
    torch.backends.cudnn.benchmark = not b


def get_root_dir() -> str:
    """get root directory of sfb package"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


def find_file(fname: str) -> Union[str, None]:
    found = None
    d = get_root_dir()
    N = [fname, os.path.join(d, fname)] + list(map(lambda x: os.path.join(d, x, fname), ["train", "sfb", "utils"]))
    for f in N:
        if os.path.exists(f):
            found = f
            break
    return found


def daydate():
    s = datetime.datetime.now()
    return s.strftime("%Y%m%d")


def move_if_exists(filename: str) -> bool:
    """
    move file if it already exists, appends _k (k natural number smaller 1000) to filename
    """
    if not os.path.exists(filename):
        return False
    oldname = filename
    pre = filename.split(".")[:-1]
    end = filename.split(".")[-1]
    if len(pre) > 1:
        p = ".".join(pre)
    else:
        p = pre[0]
    k = 1
    while os.path.exists(filename):
        filename = f"{p}_{k}.{end}"
        k += 1
        if k > 1e3:
            raise (ValueError("Too many file exists. Can not move {}".format(filename)))
    shutil.move(oldname, filename)
    return True


def save_config(obj, logdir: str) -> bool:
    """
    Save the dictionay :code:`obj` inside `config.json` in :code:`logdir`
    """
    if len(obj.keys()) == 0:
        return False
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    fname = os.path.join(logdir, "config.json")
    move_if_exists(fname)
    if "creation_time" not in obj.keys():
        obj["creation_time"] = time.strftime("%Y-%m-%d")
    if "commit" not in obj.keys():
        try:
            s = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
            obj["commit"] = s
        except:
            pass

    with open(fname, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return True


def swap_tld(f: str, new: str) -> str:
    """
    Change the file ending of filename :code:`f` to :code:`new`
    """
    if len(f) < 2 or "." not in f:
        return f
    sf = f.split(".")
    return ".".join(sf[:-1] + [new])


def fix_weights(d: OrderedDict) -> OrderedDict:
    """
    Fix weights dict, since DistributedDataParallel wraps the network torch.nn.Module inside a :code:`module`
    and thus it can no longer be read using the original one.
    """
    w: OrderedDict = OrderedDict()
    for k in d.keys():
        if "module" in k:
            kk = ".".join(k.split(".")[1:])
        else:
            kk = k
        w[kk] = d[k]
    return w


def load_weights(filename: str, device=None) -> OrderedDict:
    """
    Load weights from file. It will assume that it is a checkpoint if 'ckpt' or 'check' is in its name.
    Then it will try to extract the model/network weights from it.
    """
    if device is None:
        device = torch.device("cpu")
    if "ckpt" in filename or "check" in filename:
        data_dict = torch.load(filename, map_location=device)
        keys = data_dict.keys()
        found = False
        for k in ["state_dict", "model", "models", "net", "network", "networks"]:
            if k in keys:
                data = data_dict[k]
                found = True
                break
        if not found:
            raise KeyError("no weights found in %s" % filename)
    else:
        data = torch.load(filename, map_location=device)
    if isinstance(data, list):
        data = data[0]
    if isinstance(data, OrderedDict):
        return fix_weights(data)
    else:
        return data


def to_tensor(x: Union[np.ndarray, torch.Tensor], device=None) -> torch.Tensor:
    """Converts :code:`x` into a torch.Tensor"""
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if device is not None:
        return x.to(device)
    else:
        return x


def to_tensors(x: Union[List[Any], torch.Tensor, np.ndarray, dict], device=None, unpack=True):
    """Same as :func:`to_tensor` but works also on lists or dicts"""
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        return to_tensor(x, device)
    if len(x) == 1 and unpack:
        return to_tensor(x[0], device)
    if isinstance(x, dict):
        for k in x.keys():
            x[k] = to_tensor(x[k], device)
        return x
    for k in range(len(x)):
        x[k] = to_tensor(x[k], device)
    return x


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x


def to_o3d_pc(x: Union[torch.Tensor, np.ndarray]) -> o3d.geometry.PointCloud:
    """
    Convert a tensor/ndarray into a (single) open3d pointcloud
    input needs to be of shape (...*, 3)
    """
    x = to_numpy(x)
    x = x.squeeze()
    if x.ndim == 3:
        x = x.reshape(-1, 3)
    assert len(x.shape) == 2 and x.shape[1] == 3
    v = o3d.utility.Vector3dVector(x)
    return o3d.geometry.PointCloud(v)


def load_pointcloud(path: str, return_type="np", dtype=np.float32, device=None, shuffle=False):
    """
    Load pointcloud. Tries to support all formats (guesses the used format from the filename).

    :code:`return_type` can be "np" for numpy array, "pt" for pytorch tensor or "o3d" for open3d pointcloud
    """
    assert os.path.isfile(path)

    is_o3d = False
    if path.split(".")[-1] in ["blosc", "blz"]:
        data = load_blz(path, dtype=dtype).reshape(-1, 3).copy()
    elif path.split(".")[-1] == "raw":
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=dtype).reshape(-1, 3).copy()
    else:
        data = o3d.io.read_point_cloud(path)
        is_o3d = True
        data = np.asarray(data.points)
        if shuffle:
            idx = np.arange(len(data))
            np.random.shuffle(idx)
            data = data[idx]
        if return_type in ["o3d", "cloud", "open3d"]:
            data = to_o3d_pc(data)

    if return_type in ["o3d", "cloud", "open3d"]:
        if is_o3d:
            return data
        else:
            return to_o3d_pc(data)
    elif return_type in ["tensor", "torch", "pt"] or device is not None:
        pc = to_tensor(data, device=device)
    else:
        pc = data

    if shuffle:
        idx = np.arange(len(pc))
        np.random.shuffle(idx)
        pc = pc[idx]
    return pc


def save_blz(fname: str, x: np.ndarray):
    """
    compresses x using blosc and writes it to a binary file fname
    note that x gets flattened during saving i.e. multi-dimensionality gets lost
    """
    n = np.dtype(x.dtype).itemsize
    blzpacked = blosc.compress(x.tobytes(), typesize=n, cname="lz4hc")
    with open(fname, "wb") as f:
        f.write(blzpacked)


def load_blz(fname: str, dtype=np.float32) -> np.ndarray:
    """loads files save by :func:`save_blz`"""
    with open(fname, "rb") as f:
        x = blosc.decompress(f.read())
    return np.frombuffer(x, dtype)


def padsize(kernel_size: Union[List[int], int] = 3, mode: str = "same", dilation: Union[List[int], int] = 1):
    """
    translates mode to size of padding
    """
    if not isinstance(kernel_size, list):
        k = [kernel_size, kernel_size]
    else:
        k = kernel_size
    if not isinstance(dilation, list):
        d = [dilation, dilation]
    else:
        d = dilation
    assert len(d) == len(k)

    p = [0 for _ in range(len(k))]
    if mode == "same":
        for i in range(len(p)):
            p[i] = (d[i] * (k[i] - 1)) // 2

    if np.unique(p).shape[0] == 1:
        return p[0]
    return p


def get_logger(root_dir, model):
    v = "0"
    graph = hasattr(model, "example_input_array")
    return TensorBoardLogger(save_dir=root_dir, version=v, name="lightning_logs", log_graph=graph)


def get_callbacks(
    logdir: str,
    num_epochs: int,
    obs_loss: str = "val_loss",
    num_ckpt: int = 10,
    lr_scheduler: bool = False,
    early_stopping: bool = False,
    pc_save_data=None,
    **kwargs,
) -> List[Any]:
    best_weights = ModelCheckpoint(
        dirpath=logdir, filename="weights-{epoch}", save_top_k=10, save_weights_only=True, mode="min", monitor=obs_loss
    )
    checkpoints = ModelCheckpoint(
        dirpath=logdir, filename="ckpt-{epoch}", save_top_k=-1, mode="min", every_n_epochs=num_epochs // num_ckpt
    )
    callbacks = [best_weights, checkpoints]
    if lr_scheduler:
        lr_monitor = LearningRateMonitor(log_momentum=True)
        callbacks.append(lr_monitor)
    if early_stopping:
        es = EarlyStopping(obs_loss, mode="min", **kwargs)
        callbacks.append(es)
    if pc_save_data is not None:
        spc = O3DPCCallback(pc_save_data)
        callbacks.append(spc)
    return callbacks


def get_lr_scheduler(optimizer, config={}):
    """
    Create torch.optim.lr_scheduler using :code:`config` (dictionary) returns :code:`None` by default.
    Possible configuration keys are:

    * lr_scheduler: select scheduler, supports 'step', 'exponential', 'plateau'
    * lr_decay: lr decay factor (default: 0.5)
    * lr_scheduler_steps: the step size for StepLR and patience for ReduceLROnPlateau
    """
    sched_type = config.get("lr_scheduler", None)
    if sched_type is None or not sched_type:
        return None
    gamma = config.get("lr_decay", 0.5)
    n = config.get("lr_scheduler_steps", config.get("lr_scheduler_step_size", 50))
    min_lr = config.get("min_lr", 1e-5)
    reduce_on_plateau = False
    if sched_type.lower() == "step":
        scheduler = StepLR(optimizer, n, gamma=gamma)
    elif sched_type.lower() in ["exp", "exponential"]:
        scheduler = ExponentialLR(optimizer, gamma)
    elif "cos" in sched_type.lower():
        T_max = config.get("T_max", config.get("Tmax", 10))
        scheduler = CosineAnnealingLR(optimizer, T_max)
    elif "reduce" in sched_type.lower():
        scheduler = ReduceLROnPlateau(optimizer, factor=gamma, patience=n, cooldown=1, min_lr=min_lr)
        reduce_on_plateau = True
    if reduce_on_plateau:
        return {"scheduler": scheduler, "reduce_on_plateau": reduce_on_plateau, "interval": "epoch"}
    else:
        return {"scheduler": scheduler, "interval": "epoch"}


def to_dataloader(data, batch_size: int, **kwargs) -> torch.utils.data.DataLoader:
    n = kwargs.get("num_workers", 4)
    dl = kwargs.get("drop_last", True)
    dlargs = {"num_workers": n, "pin_memory": torch.cuda.is_available(), "batch_size": batch_size, "drop_last": dl}
    return torch.utils.data.DataLoader(data, **dlargs)


class O3DPCCallback(Callback):
    def __init__(self, data=None, num_steps=10):
        if isinstance(data, torch.utils.data.DataLoader):
            self.pcs = next(iter(data))
        elif isinstance(data, torch.Tensor):
            self.pcs = data
        else:
            raise ValueError("Incorrect data got {}".format(data))

        self.num_steps = num_steps

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.num_steps == 0 and trainer.is_global_zero:
            with torch.no_grad():
                out = pl_module(self.pcs.to(pl_module.device))
                out = {"vertex_positions": out.cpu()}
                writer = pl_module.logger.experiment
                writer.add_3d("outputs", out, step=trainer.current_epoch)
        if trainer.current_epoch == 0 and trainer.is_global_zero:
            x = {"vertex_positions": self.pcs.cpu()}
            writer.add_3d("inputs", x, step=trainer.current_epoch)

    def state_dict(self):
        return {"pointclouds": self.pcs}
