#!/usr/bin/env python
import os

from setuptools import setup
from setuptools import find_packages

here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "ms2query", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="ms2query",
    version=version["__version__"],
    description="Tool to query MS/MS spectra against GNPS library data.",
    long_description=readme,
    author="Netherlands eScience Center",
    author_email="",
    url="https://github.com/iomega/ms2query",
    packages=find_packages(),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    test_suite="tests",
    python_requires='>=3.7',
    install_requires=[
        "matchms>=0.6.1",
        "numpy",
        "spec2vec",
        "h5py<3.0.0",
        "tensorflow",
        "ms2deepscore",
        "gensim>=4.0.0",
        "pandas>=1.2.5"
    ],
    extras_require={"dev": ["bump2version",
                            "isort>=4.2.5,<5",
                            "prospector[with_pyroma]",
                            "pytest",
                            "pytest-cov",
                            "sphinx>=3.0.0,!=3.2.0,<4.0.0",
                            "sphinx_rtd_theme",
                            "sphinxcontrib-apidoc",
                            "yapf",],
    }
)
