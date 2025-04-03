from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open("readme.md", encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements file
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="vinkeljernet",
    version="0.1.0",
    description="Journalistic angle generator based on editorial DNA profiles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vinkeljernet Team",
    packages=find_packages(include=["vinkeljernet", "vinkeljernet.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.5.0",
            "ipython>=8.14.0",
        ],
        "docs": [
            "Sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vinkeljernet=vinkeljernet.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: Danish",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
    package_data={
        "vinkeljernet": ["templates/*", "config/*"],
    },
)