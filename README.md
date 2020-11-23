![GitHub Workflow Status](https://img.shields.io/github/workflow/status/iomega/ms2query/CI%20Build)
![GitHub](https://img.shields.io/github/license/iomega/ms2query)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8F%20%20%E2%97%8B-orange)](https://fair-software.eu)

# ms2query
MS2Query - machine learning assisted library querying of MS/MS spectra.

## Prepare environmnent
We recommend to create an Anaconda environment with

```
conda create --name ms2query python=3.7
conda activate ms2query
```
And then install the required dependencies, e.g. by running the following from the cloned directory
```
pip install -e
```
  
## Run app locally
Enter in terminal:
```
streamlit run ms2query_app.py
```
