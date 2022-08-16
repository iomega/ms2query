[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/iomega/ms2query/CI%20Build)](https://github.com/iomega/ms2query/actions/workflows/CI_build.yml)
![GitHub](https://img.shields.io/github/license/iomega/ms2query)
[![PyPI](https://img.shields.io/pypi/v/ms2query)](https://pypi.org/project/ms2query/)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

# MS2Query - Reliable and fast MS/MS spectral-based analogue search 

## The preprint is out now and can be found on: https://www.biorxiv.org/content/10.1101/2022.07.22.501125v1

MS2Query is able to search for both analogues and exact matches in large libraries. 

Metabolomics of natural extracts remains hampered by the grand challenge of metabolite annotation and identification. Only a tiny fraction of all metabolites has an annotated spectrum in a library; hence, when searching only for exact library matches such an approach generally has a low recall. An attractive alternative is searching for so-called analogues as a starting point for structural annotations; analogues are library molecules which are not exact matches, but display a high chemical similarity. However, current analogue search implementations are not very reliable yet and are relatively slow. Here, we present MS2Query, a machine learning-based tool that integrates mass spectrum-based chemical similarity predictors (Spec2Vec and MS2Deepscore) as well as detected precursor masses to rank potential analogues and exact matches. The reliability and scalability of MS2Query are encouraging steps toward higher-throughput large-scale untargeted metabolomics workflows. This offers entirely new opportunities for further increasing the annotation rate of complex metabolite mixtures.

<img src="https://github.com/iomega/ms2query/blob/main/images/workflow_ms2query.png" width="1000">


### Workflow
MS2Query is a tool for MSMS library matching, searching both for analogues and exact matches in one run. The workflow for running MS2Query first uses MS2Deepscore to calculate spectral similarity scores between all library spectra and a query spectrum. By using pre-computed MS2Deepscore embeddings for library spectra, this full-library comparison can be computed very quickly. The top 2000 spectra with the highest MS2Deepscore are selected. In contrast to other analogue search methods, no preselection on precursor m/z is performed. MS2Query optimizes re-ranking the best analogue or exact match at the top by using a random forest that combines 5 features. The random forest predicts a score between 0 and 1 between each library and query spectrum and the highest scoring library match is selected. By using a minimum threshold for this score, unreliable matches can be filtered out.

### Used features
As input for the random forest model, MS2Query uses 5 different features, calculated between the query spectrum and each of the 2000 preselected library spectra. The features are Spec2Vec, query precursor m/z, precursor m/z difference, an average MS2Deepscore over 10 chemically similar library molecules, and the average Tanimoto score for these 10 chemically similar library molecules. 

<img src="https://github.com/iomega/ms2query/blob/main/images/features_used.png" width="300"> 

For the last two features, multiple library spectra are used to calculate an average MS2Deepscore for 10 chemical similar library molecules. These library molecules are selected based on the known chemical structures of the spectra in the library. First the molecule belonging to 1 of the 2000 preselected library spectra is selected, followed by selecting the 10 library molecules that are chemically most similar. The chemical similarity is calculated by calculating a Tanimoto score between the InChi’s of the molecules in the library. For each of the 10 library molecules all corresponding library spectra are selected, which are often multiple spectra. The MS2Deepscore between these library spectra and the query spectrum is calculated and the average per library structure is taken. As an input feature for the random forest model, the average over the MS2Deepscore for the 10 library structures is used. In addition, the average of the Tanimoto score between the starting library structure and the 10 library structures is used as an additional input feature. 

<img src="https://github.com/iomega/ms2query/blob/main/images/worflow_average_ms2deepscore.png" width="700"> 

### Preliminary results
MS2Query can reliably predict good analogues and exact library matches. We demonstrate that MS2Query is able to find reliable analogues for 35% of the mass spectra during benchmarking with an average Tanimoto score of 0.67 (chemical similarity). For the benchmarking test set, any exact library matches were purposely removed from the reference library, to make sure the best possible match is an analogue. This is a large improvement compared to a modified cosine score based method, which resulted in an average Tanimoto score of 0.45 with settings that resulted in a recall of 35% on the same test set. The workflow of MS2Query is fully automated and optimized for scalability and speed. This makes it possible to run MS2Query on 1000 query spectra against a library of over 300.000 spectra in less than 15 minutes on a normal laptop. The scalability of MS2Query is an encouraging step toward higher-throughput large-scale untargeted metabolomics workflows, thereby creating the opportunity to develop entirely novel large-scale full sample comparisons. The good performance for larger molecules offers a lot of new opportunities for further increasing the annotation rate of complex metabolite mixtures, in particular for natural product relevant mass ranges. Finally, MS2Query is provided as a tested, open source Python library which grants easy access for researchers and developers. 


A preprint will be released soon with more detailed information! 

For questions regarding MS2Query you can contact niek.dejonge@wur.nl


## Documentation for users
### Prepare environmnent
We recommend to create an Anaconda environment with

```
conda create --name ms2query python=3.8
conda activate ms2query
```
### Pip install MS2Query
MS2Query can simply be installed by running:
```
pip install ms2query
```

### Run MS2Query
Below you can find an example script for running MS2Query.
Before running the script, replace the variables `ms2query_library_files_directory` and `ms2_spectra_directory` with the correct directories.

This script will first download files for a default MS2Query library.
This default library is trained on the [GNPS library](https://gnps.ucsd.edu/) from 2021-15-12.
Automatic downloading can take long, alternatively all model files can be manually downloaded from https://zenodo.org/record/6997924#.YvuonS5BxPY When manually downloading, store all these files in one directory and set ms2query_library_files_directory to this directory.

After downloading, a **library search** and an **analog search** is performed on the query spectra in your directory (`ms2_spectra_directory`).
The results generated by MS2Query, are stored as csv files in a results directory within the same directory as your query spectra.

Note: When running, Tensorflow often raises a few warnings on most computers. These warnings are raised when no GPU is installed and can be ignored. 

```python
from ms2query.run_ms2query import download_default_models, default_library_file_base_names, run_complete_folder
from ms2query.ms2library import create_library_object_from_one_dir

# Set the location where all your downloaded model files are stored
ms2query_library_files_directory = "./ms2query_library_files"
# Define the folder in which your query spectra are stored.
# Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object. 
ms2_spectra_directory = "specify_directory"

# Downloads pretrained models and files for MS2Query (>10GB download)
download_default_models(ms2query_library_files_directory, default_library_file_base_names())

# Create a MS2Library object
ms2library = create_library_object_from_one_dir(ms2query_library_files_directory, default_library_file_base_names())

# Run library search and analog search on your files.
run_complete_folder(ms2library, ms2_spectra_directory)

```
## Documentation for developers
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
To run all unit tests, to check if everything was installed successfully run: 
```
pytest
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
