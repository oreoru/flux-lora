"""
COAT Implementation for FLUX LoRA Training
Setup script
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="coat-flux-lora",
    version="0.1.0",
    author="COAT Implementation Team",
    description="COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training - FLUX LoRA Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coat-flux-lora",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.30.0",
        "diffusers>=0.20.0",
        "accelerate>=0.20.0",
        "peft>=0.5.0",
        "safetensors>=0.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "benchmark": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "coat-benchmark=benchmark_coat:main",
            "coat-integrate=ai_toolkit_integration.integrate_coat:main",
        ],
    },
)




