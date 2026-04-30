"""Setup script for conway_izh package."""

from setuptools import setup, find_packages

setup(
    name="conway_izh",
    version="0.1.0",
    description="Conway Game of Life + Izhikevich Neuron Hybrid Simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pytest>=7.0.0",
        "imageio>=2.9.0",
        "torch>=2.0.0",
    ],
)

