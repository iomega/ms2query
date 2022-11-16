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

### Preliminary results
MS2Query can reliably predict good analogues and exact library matches. We demonstrate that MS2Query is able to find reliable analogues for 35% of the mass spectra during benchmarking with an average Tanimoto score of 0.67 (chemical similarity). For the benchmarking test set, any exact library matches were purposely removed from the reference library, to make sure the best possible match is an analogue. This is a large improvement compared to a modified cosine score based method, which resulted in an average Tanimoto score of 0.45 with settings that resulted in a recall of 35% on the same test set. The workflow of MS2Query is fully automated and optimized for scalability and speed. This makes it possible to run MS2Query on 1000 query spectra against a library of over 300.000 spectra in less than 15 minutes on a normal laptop. The scalability of MS2Query is an encouraging step toward higher-throughput large-scale untargeted metabolomics workflows, thereby creating the opportunity to develop entirely novel large-scale full sample comparisons. The good performance for larger molecules offers a lot of new opportunities for further increasing the annotation rate of complex metabolite mixtures, in particular for natural product relevant mass ranges. Finally, MS2Query is provided as a tested, open source Python library which grants easy access for researchers and developers. 

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
Automatic downloading can take long, alternatively all model files can be manually downloaded from https://zenodo.org/record/6997924#.YvuonS5BxPY for positive mode and https://zenodo.org/record/7107654#.Yy3BeKRBxPY for negative mode. When manually downloading, store all these files in one directory and set ms2query_library_files_directory to this directory.

After downloading, a **library search** and an **analog search** is performed on the query spectra in your directory (`ms2_spectra_directory`).
The results generated by MS2Query, are stored as csv files in a results directory within the same directory as your query spectra.

Note: When running, Tensorflow often raises a few warnings on most computers. These warnings are raised when no GPU is installed and can be ignored. 

```python
from ms2query.run_ms2query import download_zenodo_files, run_complete_folder
from ms2query.ms2library import create_library_object_from_one_dir

# Set the location where downloaded library and model files are stored
ms2query_library_files_directory = "./ms2query_library_files"

# Define the folder in which your query spectra are stored.
# Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object. 
ms2_spectra_directory = 
ion_mode = # Fill in "positive" or "negative" to indicate for which ion mode you would like to download the library

zenodo_DOIs = {"positive": 6997924, 
               "negative": 7107654}

# Downloads pretrained models and files for MS2Query (>2GB download)
download_zenodo_files(zenodo_DOIs[ion_mode], 
                      ms2query_library_files_directory)

# Create a MS2Library object
ms2library = create_library_object_from_one_dir(ms2query_library_files_directory)

# Run library search and analog search on your files.
run_complete_folder(ms2library, ms2_spectra_directory)

```

## Create your own library (without training new models)
The code below creates all required library files for your own in house library. 
No new models for MS2deepscore, Spec2Vec and MS2Query will be trained, to do this see the next section.

To create your own library you also need to install RDKit, by running the following in your command line (while in the ms2query conda environment):
```
conda install -c conda-forge rdkit
```

It is important that the library spectra are annotated with smiles, inchi's or inchikeys in the metadata otherwise they
are not included in the library. 

Fill in the blank spots with the file locations.
The models for spec2vec, ms2deepscore and ms2query can be downloaded from the zenodo links (see above).

```python
from ms2query.create_new_library.library_files_creator import LibraryFilesCreator
from ms2query.clean_and_filter_spectra import clean_normalize_and_split_annotated_spectra
from ms2query.utils import load_matchms_spectrum_objects_from_file

spectrum_file_location =  # The file location of the library spectra
library_spectra = load_matchms_spectrum_objects_from_file(spectrum_file_location)
# Fill in the missing values:
cleaned_library_spectra = clean_normalize_and_split_annotated_spectra(library_spectra, ion_mode_to_keep="")[
    0]  # fill in "positive" or "negative"
library_creator = LibraryFilesCreator(cleaned_library_spectra,
                                      output_directory="",  # For instance "data/library_data/all_GNPS_positive_mode_"
                                      ms2ds_model_file_name="",  # The file location of the ms2ds model
                                      s2v_model_file_name="", )  # The file location of the s2v model
library_creator.create_all_library_files()
```

To run MS2Query on your own created library run (again fill in the blanks).
For the SQLite file, S2V embeddings file and the ms2ds embeddings file. The files just calculated by you should be used.
For the other files the files downloaded from zenodo (see "run MS2Query") should be used. 
The results will be returned as csv files in a results directory. 
```python
from ms2query.run_ms2query import run_complete_folder
from ms2query.ms2library import create_library_object_from_one_dir

# Define the folder in which your query spectra are stored.
# Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object. 
ms2_spectra_directory = 
ms2_library_directory = # Specify the directory containing all the library and model files

# Create a MS2Library object from one directory
# If this does not work (because files have unexpected names or are not in one dir) see below.
ms2library = create_library_object_from_one_dir(ms2_library_directory)

# Run library search and analog search on your files.
run_complete_folder(ms2library, ms2_spectra_directory)
```

An alternative for loading in a ms2library is by specifying each file type manually, which is needed if not all file are in one dir or if the library files or models have unexpected names

```python
from ms2query.ms2library import MS2Library
# Specify all the file locations 
ms2library = MS2Library(sqlite_file_name= ,
                        s2v_model_file_name= ,
                        ms2ds_model_file_name= ,
                        pickled_s2v_embeddings_file_name= ,
                        pickled_ms2ds_embeddings_file_name= ,
                        ms2query_model_file_name= ,
                        classifier_csv_file_name= , #Leave None if not available
                        )
```

# Create your own library and train new models
The code trains new MS2Deepscore, Spec2Vec and MS2Query models for your in house library, 
and creates all needed files for running MS2Query. 

It is important that the library spectra are annotated with smiles, inchi's or inchikeys in the metadata otherwise they
are not included in the library and training. 

Fill in the blank spots below and run the code (can take several days). 
The models will be stored in the specified output_folder. MS2Query can be run

```python
from ms2query.create_new_library.train_models import clean_and_train_models
clean_and_train_models(spectrum_file=, #Fill in the location of the file containing the library spectra
                       # Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object. 
                       ion_mode=, # Fill in the ion mode, choose from "positive" or "negative"
                       output_folder= # The output folder in which all the models are stored. 
                       )
```

To run MS2Query on your own created library run the code below (again fill in the blanks).

```python
from ms2query.run_ms2query import run_complete_folder
from ms2query.ms2library import create_library_object_from_one_dir

# Define the folder in which your query spectra are stored.
# Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object. 
ms2_spectra_directory = # Specify the folder containing the query spectra you want to run against the library
ms2_library_directory = # Specify the directory containing all the library and model files

# Create a MS2Library object from one directory
# If this does not work (because files have unexpected names or are not in one dir) see below.
ms2library = create_library_object_from_one_dir(ms2_library_directory)

# Run library search and analog search on your files.
run_complete_folder(ms2library, ms2_spectra_directory)
```

After running the model can be loaded 

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
