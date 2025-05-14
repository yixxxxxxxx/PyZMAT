# setup.py
from setuptools import setup, find_packages

setup(
    name="pyzmat",
    version="0.1.0",
    description="Wrapper around ASE & ML-FFs for internal-coordinate workflows",
    author="Yixuan Huang",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ase>=3.23.1",
        "numpy>=1.20",
        "scipy>=1.6"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)