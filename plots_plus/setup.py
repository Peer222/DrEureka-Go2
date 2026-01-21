from setuptools import find_packages
from distutils.core import setup

setup(
    name='plots_plus',
    version='1.0.0',
    author='Peer Duensing',
    license="BSD-3-Clause",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    author_email='p.duensing@stud.uni-hannover.de',
    install_requires=[
        "seaborn", "pandas", "tyro", "numpy"
    ]
)
