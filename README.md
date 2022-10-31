# pointcloudmatcher

Code for the paper: "Cluster-Based Autoencoders for Volumetric Point Clouds".

## Installation

Create a python virtual environment and [pytorch](https://pytorch.org/get-started/locally/) (preferably version 1.11) and [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)  (tested with version 0.6.2).

Then one also needs to install the following packages manually since we used non-release versions for them.

* **Geomloss**
It is best to install the version from [pull#49](https://github.com/jeanfeydy/geomloss/pull/49) since 
it contains some performance optimizations for evaluation.
```
git clone https://github.com/jeanfeydy/geomloss && cd geomloss
git remote add 2origin https://github.com/martenlienen/geomloss
git pull 2origin
git checkout e6ff657d4a063bfb08351c08fc3e11eb55d0c319
pip install -e .
```

* **kmeans-pytorch**
Install the latest version via git i.e. `pip install 'git+https://github.com/subhadarship/kmeans_pytorch'`
or to get the exact version
```
git clone https://github.com/subhadarship/kmeans_pytorch && cd kmeans_pytorch
git checkout f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805
pip install -e .
```

* **k-means-constrained**
```
pip install 'numpy<1.23'
pip  install  --no-build-isolation k-means-constrained==0.6.0 --no-binary k-means-constrained
```

Then install the rest with
```
pip install -e .
```

## Usage

```python
import numpy as np
import sfb

# load pretrained model
model = sfb.load_network("weights/foldingnet.ckpt")

# load pointcloud
x = ...

# define interpolation factor
alpha = 0.2

design_pc = sfb.density.normalize_design_pc(np.loadtxt("pointclouds/stripe.xyz"))

z = sfb.apply_design(model, x, design_pc, model.config["num_points"], alpha)
```

## Training

In order to train a model with your own dataset one first needs to apply the design point clouds (optional)
and then cluster the pointclouds (unless they are already of the desired size).
We provide a function to preprocess the data `preparedata` which works for xyz point clouds.
If on has xyz point clouds in the folder `/data/` and a design point cloud `/example/design.xyz`, then
use
```python
sfb.preparedata(["/data/"], ["/example/design.xyz"], "/output/", 2048)
```
This will create folders `/output/applied/design` with `design.xyz` applied to the point clouds
and `/output/combined_split` with the individual clusters of all point clouds (in binary format).

Then one can train a model
```python
python train.py --num_points=2048 --emd --blosc --datroot=/output/combined_split --factors_file=weights/factors.json
```
See `python train.py --help` for additional options.
