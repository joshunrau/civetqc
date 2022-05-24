import os

from pathlib import Path
from pkg_resources import parse_requirements
from setuptools import find_packages, setup


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PACKAGE_DIR = os.path.join(BASE_DIR, "civetqc")

def get_long_description():
    with open(os.path.join(BASE_DIR, "README.md"), "r") as file:
        return file.read()

def get_install_requires():
    with Path('requirements.txt').open() as requirements_txt:
        return [str(r) for r in parse_requirements(requirements_txt)]

setup(
    name = "civetqc",
    version = "0.0.4",
    author="Joshua Unrau",
    author_email="contact@joshuaunrau.com",
    description="civetqc is a command-line utility for automated quality control of CIVET outputs",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/joshunrau/civetqc",
    project_urls={
        "Bug Tracker": "https://github.com/joshunrau/civetqc/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="."),
    python_requires=">=3.9",
    install_requires = [
        'matplotlib==3.5.2', 
        'numpy==1.22.4', 
        'pandas==1.4.2', 
        'scikit_learn==1.1.1', 
        'scikit_optimize==0.9.0',
        'seaborn==0.11.2',
        'setuptools==62.3.2',
        'yellowbrick==1.4'
    ],
    entry_points={
        'console_scripts': [
            'civetqc=civetqc.main:main'
        ]
    },
    include_package_data=True,
    package_data = {
        "" : [
            "resources/*.json",
            "resources/saved_models/*.pkl", 
            "resources/simulated_data/*.csv",
        ]
    }
)

