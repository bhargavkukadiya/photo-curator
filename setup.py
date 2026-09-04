#!/usr/bin/env python3
"""Setup configuration for photo-curator."""

from setuptools import find_packages, setup

setup(
    name="photo-curator",
    version="1.1.0",
    description="AI-powered photo selector that curates, ranks, and deduplicates photo collections.",
    author="Bhargav Kukadiya",
    url="https://github.com/bhargavkukadiya/photo-curator",
    package_dir={"photo_curator": "src/photo_curator"},
    packages=["photo_curator"],
    py_modules=["album_selector"],
    entry_points={
        "console_scripts": [
            "photo-curator = photo_curator.cli:main",
        ],
    },
)
