from setuptools import setup, find_packages

setup(
    name="pepper",
    version="0.0.1",
    url="https://gitlab.cern.ch/desy-ttbarbsm-coffea/pepper/",
    description="""
A python framework for analyzing NanoAODs.
Easy to use and highly configurable.
""",
    packages=find_packages(),
    install_requires=[
        "coffea",
        "awkward>=1.2",
        "parsl",
        "h5py",
        "hdf5plugin",
        "hjson",
    ],
)
