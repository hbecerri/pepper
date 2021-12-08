from setuptools import setup, find_packages

setup(
    name="pepper",
    version="0.0.1",
    url="https://gitlab.cern.ch/pepper/pepper/",
    description="""
A python framework for analyzing NanoAODs.
Easy to use and highly configurable.
""",
    packages=find_packages(),
    install_requires=[
        "coffea>=0.7.11",
        "awkward>=1.7.0",
        "parsl>=1.1",
        "h5py",
        "hdf5plugin",
        "hist",
        "hjson",
    ],
)
