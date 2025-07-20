#!/usr/bin/env python3
"""
Setup script for UAV Landing System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="uav_landing_system",
    version="1.0.0",
    description="Production-ready UAV landing system with neurosymbolic memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MegaPanchamZ",
    author_email="NOTGONAGIVEITTAYA@gmail.com",
    url="https://github.com/MegaPanchamZ/uav_landing_project",
    
    packages=find_packages(),
    python_requires=">=3.8",
    
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "onnxruntime>=1.10.0",  # CPU version
        # "onnxruntime-gpu>=1.10.0",  # Uncomment for GPU support
    ],
    
    extras_require={
        "gpu": ["onnxruntime-gpu>=1.10.0"],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "uav-landing=uav_landing_main:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    
    keywords="uav drone landing computer-vision deep-learning onnx neurosymbolic",
    project_urls={
        "Documentation": "https://github.com/MegaPanchamZ/uav_landing_project/docs",
        "Source": "https://github.com/MegaPanchamZ/uav_landing_project",
        "Tracker": "https://github.com/MegaPanchamZ/uav_landing_project/issues",
    },
)
