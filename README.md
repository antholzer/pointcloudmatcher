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
