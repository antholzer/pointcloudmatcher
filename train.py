import os
import argparse
import torch
import sfb
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from sfb.data import xyzData, BloscData
from sfb.data import RandomRotation, Normalize, max_emd_chamfer_distances


def main(config):
    config["seed"] = torch.randint(2**32 - 1, (1,)).item()
    sfb.seed(config["seed"])

    if config.get("lr_scheduler", False):
        config["lr_scheduler"] = "reduce"
    logdir = sfb.utils.get_logdir("foldingnet", config)
    config["name"] = "foldingnet"
    datadir = config.get("dataroot", config.get("datadir", None))
    if datadir is None or not os.path.exists(datadir):
        raise ValueError("data directory not found")
    num_points = config.get("num_points")

    if config.get("xyz", False):
        transform = torch.nn.Sequential(Normalize(), RandomRotation())
        train_data = xyzData(datadir, num_points=num_points, split=0.8, transform=transform)
        test_data = xyzData(datadir, num_points=num_points, split=0.8, train=False, transform=transform)
    elif config.get("blosc", False):
        transform = RandomRotation()
        train_data = BloscData(datadir, num_points=num_points, split=0.8, transform=transform)
        test_data = BloscData(datadir, num_points=num_points, split=0.8, train=False, transform=transform)
    else:
        raise ValueError("Dataloader not defined pass --xyz or --blosc")

    if len(config["factors_file"]) > 0:
        c = "dataroot: {}\nseed: {}".format(config["dataroot"], config["seed"])
        config["loss_factors"] = max_emd_chamfer_distances(config["factors_file"], train_data, 0.02, comment=c)

    sfb.utils.save_config(config, logdir)

    model = sfb.FoldingNet(**config)
    if config["init"] is not None:
        weights = sfb.utils.load_weights(config["init"])
        model.load_state_dict(weights)
        print("Initialized with weights {}".format(config["init"]))

    train_loader = sfb.models.to_dataloader(train_data, config["batch_size"])
    test_loader = sfb.models.to_dataloader(test_data, config["batch_size"])
    callbacks = sfb.models.get_callbacks(
        logdir, config["num_epochs"], lr_scheduler=config.get("lr_scheduler", None), pc_save_data=test_loader
    )

    if config["dist"]:
        trainer = pl.Trainer(
            default_root_dir=logdir,
            logger=sfb.models.get_logger(logdir, model),
            max_epochs=config["num_epochs"],
            callbacks=callbacks,
            accelerator="gpu",
            devices=torch.cuda.device_count(),
            strategy=DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=logdir,
            logger=sfb.models.get_logger(logdir, model),
            max_epochs=config["num_epochs"],
            gpus=1,
            callbacks=callbacks
        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, help="path to dataset folder")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="Number of GPUs to use. Set to -1 to use all available ones"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training/validation")
    parser.add_argument("--lr", type=float, default=0.0001, help="Starting learning rate")
    parser.add_argument("--lr_scheduler", action="store_true", help="use ReduceLROnPlateau lr_scheduler")
    parser.add_argument("--lr_scheduler_step_size", type=int, default=10, help="Step size of lr scheduler")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Min lr for lr_scheduler")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="LR scheduler decay factor")
    parser.add_argument("--name", type=str, default="", help="Add extra name to logdir")
    parser.add_argument("--early_stopping", action="store_true", help="Use early stopping")
    parser.add_argument("--es_n", type=int, default=10, help="Epoch patience for early stopping")
    parser.add_argument(
        "--es_delta",
        type=float,
        default=0.001,
        help="loss needs to improve by this much in order to be considered a change for no early stopping"
    )
    parser.add_argument("--es_stop", type=float, default=1.0, help="if loss is higher than this value stop training")
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--shape", type=str, default="cube")
    parser.add_argument("--ply", action="store_true")
    parser.add_argument("--xyz", action="store_true")
    parser.add_argument("--blosc", action="store_true")
    parser.add_argument("--num_points", type=int, default=2048, help="number of output points")
    parser.add_argument("--k", "--K", type=int, default=16, help="K in K-NN graph in encoder")
    parser.add_argument("--init", type=str, default=None, help="path to initial weights")
    parser.add_argument("--emd", action="store_true", help="use EMD as loss function")
    parser.add_argument("--p", type=float, default=2, help="use p-norm for EMD")
    parser.add_argument("--emdscaling", type=float, default=0.9, help="closer to 1 for more accurate but slower EMD")
    parser.add_argument("--factors_file", type=str, default="", help="json file with loss weight factors")
    parser.add_argument(
        "--chamfer_factor",
        type=float,
        default=1.0,
        help="Also use chamfer distance with this factor together with EMD"
    )

    args = vars(parser.parse_args())
    if not os.path.exists(args["dataroot"]):
        raise ValueError("data root does not exist")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU detected!")
        device = torch.device("cpu")
    ngpus = torch.cuda.device_count()

    if args["gpu"] >= 0 or ngpus <= 1:
        args["dist"] = False
    else:
        args["dist"] = True
    main(args)
