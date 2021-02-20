from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mutual_info",
    version="0.31",
    description="Putting GaelVaraquaux's gist in a repo. Mutual information calculation utils.",
    url="https://github.com/mutualinfo/mutual_info",
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
    ],
    packages=["mutual_info"],
)
