"""Setup script for PyTorch GammaNet."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pytorch-gammanet",
    version="0.1.0",
    author="GammaNet Contributors",
    description="PyTorch implementation of GammaNet - recurrent neural networks inspired by cortical circuits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pytorch-gammanet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "experiments": ["matplotlib", "seaborn", "pandas"],
    },
    entry_points={
        "console_scripts": [
            "gammanet-train=scripts.train:main",
            "gammanet-evaluate=scripts.evaluate:main",
        ],
    },
)