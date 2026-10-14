from setuptools import setup, find_packages
import os

def read_requirements(filename):
    """Read dependencies from a requirements file."""
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="fisheye-charuco-calibration",
    version="0.2.0",
    author="Jamie Milsom et al.",
    description="Fisheye and pinhole camera intrinsic calibration using OpenCV ChArUco boards",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jujupedro5052005/Fisheye_ChArUco_Calibration",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": ["pytest"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.10",
)
