#!/usr/bin/env python
import os
from setuptools import find_packages, setup


here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "ms2query", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="ms2query",
    version=version["__version__"],
    entry_points={"console_scripts": ["ms2query=ms2query:command_line"]},
    description="Tool to query MS/MS spectra against mass spectral library",
    long_description_content_type="text/markdown",
    long_description=readme,
    author="Netherlands eScience Center",
    author_email="",
    url="https://github.com/iomega/ms2query",
    packages=find_packages(),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    test_suite="tests",
    python_requires='>=3.8',
    install_requires=[
        "matchms>=0.24.0,<=0.26.4",
        "numpy",
        "spec2vec>=0.6.0",
        "h5py",
        "pyarrow",
        "scikit-learn",
        "ms2deepscore==0.5.0",
        "gensim>=4.0.0",
        "pandas",
        "tqdm",
        "matplotlib",
        "skl2onnx",
        "onnxruntime<1.16", # 1.16 breaks the code due to the issue https://github.com/iomega/ms2query/issues/208
    ],
    extras_require={"dev": ["bump2version",
                            "isort>=5.1.0",
                            "prospector",
                            "pytest",
                            "pytest-cov",
                            "sphinx>=3.0.0,!=3.2.0,<4.0.0",
                            "sphinx_rtd_theme",
                            "sphinxcontrib-apidoc",
                            "yapf",
                            "rdkit"],
    }
)
