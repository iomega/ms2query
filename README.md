![GitHub Workflow Status](https://img.shields.io/github/workflow/status/iomega/ms2query/CI%20Build)
![GitHub](https://img.shields.io/github/license/iomega/ms2query)
[![PyPI](https://img.shields.io/pypi/v/ms2query)](https://pypi.org/project/ms2query/)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

<img src="https://github.com/iomega/ms2query/blob/main/images/ms2query_logo.svg" width="280">

### MS2Query - machine learning assisted library querying of MS/MS spectra.
MS2Query is a tool for fast library searching for both analogs and true matches.

## Local installation of MS2Query
### Prepare environmnent
We recommend to create an Anaconda environment with

```
conda create --name ms2query python=3.7
conda activate ms2query
```
### Clone repository
Clone the present repository, e.g. by running
```
git clone https://github.com/iomega/ms2query.git
```
And then install the required dependencies, e.g. by running the following from within the cloned directory
```
pip install -e .
```

## Contributing

If you want to contribute to the development of ms2query,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## License

Copyright (c) 2021, Netherlands eScience Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
