from setuptools import setup, find_packages

required = [
    "torch>=1.9,<2", "tqdm>4,<5", "plyfile<0.8", "opencv-python>4,<5", "h5py>=3,<4", "tensorboard>2,<3",
    "pykeops>=1.5,<=2", "blosc>=1.9,<2", "kornia>=0.4", "pytorch3d>=0.6,<0.7", "pandas>=1,<2", "open3d>=0.13,<0.16",
    "scipy>=1.6,<2", "scikit-learn>=1,<2", "pytorch_lightning>1.2,<2", "kmeans_pytorch", "fastcore>=1,<2", "numba",
    "geomloss", "k-means-constrained>=0.6,<0.7", "matplotlib", "ortools<9.3"
]


setup(
    name="sfb",
    version="0.1",
    description="Code for the machine learning part of the SFB projet.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Stephan Antholzer and Martin Berger",
    author_email="stephan.antholzer@uibk.ac.at, martin.berger@uibk.ac.at",
    url="https://git.uibk.ac.at/c7021062/sfb",
    packages=find_packages(include=("sfb",)),
    license="MIT",
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.6,<4",
)
