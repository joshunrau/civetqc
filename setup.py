from setuptools import setup, find_packages


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    return long_description

setup(
    name="civetqc",
    version="0.0.1",
    author="Joshua Unrau",
    author_email="joshua.unrau@mail.mcgill.ca",
    description="CivetQC is a command-line utility for automated quality control of CIVET outputs.",
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
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    scripts=["civetqc"],
    include_package_data=True,
    package_data={"" : ["data/simulated_data/*.csv", "model/*.pkl"]}
)