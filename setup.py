"""
Setup script for MVP detector.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mvp-detector",
    version="1.0.0",
    author="Binhua Huang",
    author_email="binhua.huang@example.com",
    description="Motion Vector Propagation for Zero-Shot Object Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microa/MVP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "mvp-detect=src.main:main",
            "mvp-evaluate=evaluation.evaluate:main",
            "mvp-extract-mv=utils.extract_motion_vectors:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
