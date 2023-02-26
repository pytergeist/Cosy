import os
from setuptools import setup, find_packages

cwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cwd, "README.md")) as f:
    long_description = f.read()


setup(
    name="cosy",
    version="0.0.1",
    description="Simple keras wrappers for soft parameter sharing multitask learning",
    long_description=long_description,
    author="Tom Pope",
    author_email="tompopeworks@gmail.com",
    url="https://github.com/ThePopeLabs/REvolve",
    install_requires=[
        "numpy",
    ],
    packages=find_packages(exclude=["cosy.tests*"]),
)
