#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Setup torchal."""

from setuptools import find_packages, setup


def read_requirements():
    """Retrieves the list of packages mentioned in requirements.txt"""
    req_list = []
    with open("requirements.txt", "r") as f:
        for line in f.readlines():
            line = line.rstrip("\n")
            req_list.append(line)
    return req_list


def readme():
    """Retrieves the readme content."""
    with open("README.md", "r") as f:
        content = f.read()
    return content


setup(
    name="torchal",
    version="0.0.1",
    description="A codebase for active learning built on top of pycls.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Prateek Munjal",
    author_email="prateekmunjal31@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=read_requirements(),
    url="https://github.com/PrateekMunjal/TorchAL",
    # install_requires=["numpy", "opencv-python", "simplejson", "yacs"],
)
