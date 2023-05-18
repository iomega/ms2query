[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/iomega/ms2query/CI_build.yml)](https://github.com/iomega/ms2query/actions/workflows/CI_build.yml)
![GitHub](https://img.shields.io/github/license/iomega/ms2query)
[![PyPI](https://img.shields.io/pypi/v/ms2query)](https://pypi.org/project/ms2query/)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![DOI](https://zenodo.org/badge/306595295.svg)](https://zenodo.org/badge/latestdoi/306595295)


# MS2Query - Reliable and fast MS/MS spectral-based analogue search 

# Contents
* [Overview](https://github.com/iomega/ms2query#overview)
* [Installation guide](https://github.com/iomega/ms2query#Installation-guide)
* [Run MS2Query from command line](https://github.com/iomega/ms2query#Run-MS2Query-from-command-line)
* [Build MS2Query into other tools](https://github.com/iomega/ms2query#Build-MS2Query-into-other-tools)
* [Create your own library](https://github.com/iomega/ms2query#Create-your-own-library-without-training-new-models)
* [Train new models](https://github.com/iomega/ms2query#Create-your-own-library-and-train-new-models)
* [Documentation for developers](https://github.com/iomega/ms2query#Documentation-for-developers)
* [Recreate Results Manuscript](http://github.com/iomega/ms2query#Recreate-Results-Manuscript)
* [Contributing](https://github.com/iomega/ms2query#Contributing)
* [License](https://github.com/iomega/ms2query#License)


## Overview

**The publication can be found here: https://rdcu.be/c8Hkc
Please cite this article when using MS2Query**

MS2Query uses MS2 mass spectral data to find the best match in a library 
and is able to search for both analogues and exact matches. 
A pretrained library for MS2Query is available based on the GNPS library. 
In our benchmarking we show that MS2Query performs better compared to current standards in the field like Cosine Score and the Modified Cosine score. 
MS2Query is easy to install (see below) and is scalable to large numbers of MS2 spectra. 

<img src="https://github.com/iomega/ms2query/blob/main/images/workflow_ms2query.png" width="1000">

### Workflow
MS2Query is a tool for MSMS library matching, searching both for analogues and exact matches in one run. The workflow for running MS2Query first uses MS2Deepscore to calculate spectral similarity scores between all library spectra and a query spectrum. By using pre-computed MS2Deepscore embeddings for library spectra, this full-library comparison can be computed very quickly. The top 2000 spectra with the highest MS2Deepscore are selected. In contrast to other analogue search methods, no preselection on precursor m/z is performed. MS2Query optimizes re-ranking the best analogue or exact match at the top by using a random forest that combines 5 features. The random forest predicts a score between 0 and 1 between each library and query spectrum and the highest scoring library match is selected. By using a minimum threshold for this score, unreliable matches are filtered out.

For questions regarding MS2Query please make an issue on github or contact niek.dejonge@wur.nl


## Installation guide
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
All dependencies are automatically installed, the dependencies can be found in setup.py. 
The installation is expected to take about 2 minutes. 
MS2Query is tested by continous integration on MacOS, Windows and Ubuntu for python version 3.7 and 3.8. 

## Run MS2Query from command line

#### Download default library

When running for the first time a pretrained ms2query library should be downloaded. 
Change the file locations to the location where the library should be stored. 
Change the --ionmode to the needed ionmode (positive or negative)

```console
ms2query --library .\folder_to_store_the_library --download --ionmode positive 
```

Alternatively all model files can be manually downloaded from 
https://zenodo.org/record/6124552 for positive mode and 
https://zenodo.org/record/7104184 for negative mode.

#### Preprocessing mass spectra
MS2Query is run on all MS2 spectra in a spectrum file. 
MS2Query does not do any peak picking or clustering of similar MS2 spectra.
If your files contain many MS2 spectra per feature it is advised to first reduce the number of MS2 spectra 
by clustering or feature selection. There are multiple tools available that do this. 
One reliable method is using MZMine for preprocessing, https://mzmine.github.io/mzmine_documentation/index.html. 
As input for MS2Query you can use the MGF file of the FBMN output of MZMine, 
see https://ccms-ucsd.github.io/GNPSDocumentation/featurebasedmolecularnetworking-with-mzmine2/.

#### Running MS2Query
After downloading a default library MS2Query can be run on your MS2 spectra. 
Run the command below and specify the location where your spectra are stored. 
If a spectrum file is specified all spectra in this folder will be processed. 
If a folder is specified all spectrum files within this folder will be processed. 
The results generated by MS2Query, are stored as csv files in a results directory within the same directory as your query spectra. 

```console
ms2query --spectra .\location_of_spectra --library .\library_folder --ionmode positive 
```
To do a test run with dummy data you can download the file 
[dummy_spectra.mgf](https://github.com/iomega/ms2query/blob/main/dummy_data/dummy_spectra.mgf). 
The expected results can be found in [expected_results_dummy_data.csv](https://github.com/iomega/ms2query/blob/main/dummy_data/expected_results_dummy_data.csv). 
After downloading the library files, running on the dummy data is expected to take less than half a minute.

Run ms2query --help for more info/options, or see below:

```console
usage: MS2Query [-h] [--spectra SPECTRA] --library LIBRARY_FOLDER [--ionmode {positive,negative}] [--download] [--results RESULTS] [--filter_ionmode]

MS2Query is a tool for MSMS library matching, searching both for analogues and exact matches in one run

optional arguments:
  -h, --help            show this help message and exit
  --spectra SPECTRA     The MS2 query spectra that should be processed. If a directory is specified all spectrum files in the directory will be processed. Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a
                        pickled matchms object
  --library LIBRARY_FOLDER
                        The directory containing the library spectra, to download add --download
  --ionmode {positive,negative}
                        Specify the ionization mode used
  --download            This will download the most up to date model and library.The model will be stored in the folder given as the second argumentThe model will be downloaded in the in the ionization mode specified under --mode
  --results RESULTS     The folder in which the results should be stored. The default is a new results folder in the folder with the spectra
  --filter_ionmode      Filter out all spectra that are not in the specified ion-mode. The ion mode can be specified by using --ionmode
  --addional_metadata   Return additional metadata columns in the results, for example --additional_metadata rtinseconds feature_id
```

## Build MS2Query into other tools
If you want to incorporate MS2Query into another tool it might be easier to run MS2Query from a python script, 
instead of running from the command line. The guide below can be used as a starting point. 

Below you can find an example script for running MS2Query.
Before running the script, replace the variables `ms2query_library_files_directory` and `ms2_spectra_directory` with the correct directories.

This script will first download files for a default MS2Query library.
This default library is trained on the [GNPS library](https://gnps.ucsd.edu/) from 2021-15-12.

After downloading, a **library search** and an **analog search** is performed on the query spectra in your directory (`ms2_spectra_directory`).
The results generated by MS2Query, are stored as csv files in a results directory within the same directory as your query spectra.

```python
from ms2query.run_ms2query import download_zenodo_files, run_complete_folder
from ms2query.ms2library import create_library_object_from_one_dir

# Set the location where downloaded library and model files are stored
ms2query_library_files_directory = "./ms2query_library_files"

# Define the folder in which your query spectra are stored.
# Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object. 
ms2_spectra_directory = 
ion_mode = # Fill in "positive" or "negative" to indicate for which ion mode you would like to download the library

# Downloads pretrained models and files for MS2Query (>2GB download)
download_zenodo_files(ion_mode, 
                      ms2query_library_files_directory)

# Create a MS2Library object
ms2library = create_library_object_from_one_dir(ms2query_library_files_directory)

# Run library search and analog search on your files.
run_complete_folder(ms2library, ms2_spectra_directory)

```
## Create your own library (without training new models)
The code below creates all required library files for your own in house library. 
No new models for MS2deepscore, Spec2Vec and MS2Query will be trained, to do this see the next section.

First install MS2Query (see above under installation guide)
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
from ms2query.utils import load_matchms_spectrum_objects_from_file, select_files_in_directory
from ms2query.run_ms2query import download_zenodo_files
from ms2query.ms2library import select_files_for_ms2query


spectrum_file_location =  # The file location of your library spectra
ionisation_mode = #The ionisation mode, choose between "positive" or "negative"
directory_for_library_and_models = # Fill in the direcory in which the models will be downloaded and the library will be stored.

# Downloads the models:
download_zenodo_files(ionisation_mode, directory_for_library_and_models, only_models=True)
library_spectra = load_matchms_spectrum_objects_from_file(spectrum_file_location)

files_in_directory = select_files_in_directory(directory_for_library_and_models)
dict_with_file_names = select_files_for_ms2query(files_in_directory, ["s2v_model", "ms2ds_model", "ms2query_model"])
ms2ds_model_file_name = dict_with_file_names["ms2ds_model"]
s2v_model_file_name = dict_with_file_names["s2v_model"]
ms2query_model = dict_with_file_names["ms2query_model"]

# Fill in the missing values:
cleaned_library_spectra = clean_normalize_and_split_annotated_spectra(library_spectra, ion_mode_to_keep="")[0]

library_creator = LibraryFilesCreator(cleaned_library_spectra,
                                      output_directory=directory_for_library_and_models, 
                                      ms2ds_model_file_name=ms2ds_model_file_name, 
                                      s2v_model_file_name=s2v_model_file_name, ) 
library_creator.create_all_library_files()
```

To run MS2Query on your own created library. Check out the instructions under Run MS2Query. Both command line and the code version should work.
Make sure that the downloaded models and the SQLite file, S2V embeddings file and the ms2ds embeddings file,
just generated by you, are in the same directory.
The results will be returned as csv files in a results directory. 

An alternative for loading in a ms2library is by specifying each file type manually, which is needed if not all file are in one dir or if the library files or models have unexpected names.

```python
from ms2query.ms2library import MS2Library
from ms2query.run_ms2query import run_complete_folder

ms2_spectra_directory = # Fill in the location of your query spectra

# Specify all the file locations 
ms2library = MS2Library(sqlite_file_name= ,
                        s2v_model_file_name= ,
                        ms2ds_model_file_name= ,
                        pickled_s2v_embeddings_file_name= ,
                        pickled_ms2ds_embeddings_file_name= ,
                        ms2query_model_file_name= ,
                        classifier_csv_file_name= , #Leave None if not available
                        )
run_complete_folder(ms2library, ms2_spectra_directory)

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

After running you can run MS2Query on your newly created models and library. See above on how to run MS2Query.

## Documentation for developers
### Prepare environmnent
We recommend to create an Anaconda environment with

```
conda create --name ms2query python=3.8
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

## Recreate Results Manuscript

To recreate the results in Figure 2 in the MS2Query manuscript download the in between results from https://zenodo.org/record/7427094. 
Install MS2Query as described above (no need to download the models) and run the code below. This code should work with version 0.5.7. 
```python
from ms2query.benchmarking.create_accuracy_vs_recall_plot import create_plot

# The folder where the benchmarking results are stored. These can be downloaded from https://zenodo.org/record/7427094
base_folder = "./benchmarking_results"

create_plot(exact_matches=True, # Change to switch between the plot for the exact matches test st and the analogues test set
            positive=False, # Change to switch between the positive and negative ionization mode results
            recalculate_means=True, 
            save_figure=False, # If you want to save the figure, change to true
            base_folder=base_folder
            )
```

Above code only recreates the figures based on the already generated test results. To reproduce the test results from scratch the models have to be retrained. The test split is random and the models trained have a random component to it, so the results could vary sligtly, but the general conclusions from the results are expected to be the same.
From https://zenodo.org/record/7427094 download the ALL_GNPS_NO_PROPOGATED.mgf file to use the same starting data as used for the 20-fold cross-validation, this set was downloaded from https://gnps-external.ucsd.edu/gnpslibrary on 01-11-2022, alternatively a more recent version could be downloaded. To redo the analysis with exactly the same test split as in the Manuscript the test sets can be downloaded from https://zenodo.org/record/7427094 the training data can be constructed by combining the 19 other test sets together for each of the 20 data splits.

If you want to randomly recreate the test splits from scratch run:

```python
from ms2query.benchmarking.k_fold_cross_validation import split_and_store_annotated_unannotated, split_k_fold_cross_validation_analogue_test_set, split_k_fold_cross_validation_exact_match_test_set

spectrum_file_name = "./ALL_GNPS_NO_PROPOGATED.mgf"
split_and_store_annotated_unannotated(spectrum_file_name, ionmode="positive", output_folder="./positive_mode_data_split")
split_and_store_annotated_unannotated(spectrum_file_name, ionmode="negative", output_folder="./negative_mode_data_split")

positive_annotated_spectra = load_matchms_spectrum_objects_from_file("positive_mode_data_split/annotated_training_spectra.pickle")
negative_annotated_spectra = load_matchms_spectrum_objects_from_file("negative_mode_data_split/annotated_training_spectra.pickle")

# Run for positive mode spectra
split_k_fold_cross_validation_analogue_test_set(positive_annotated_spectra, 20, output_folder = "./positive_mode/analogue_test_sets_splits/)
split_k_fold_cross_validation_exact_match_test_set(positive_annotated_spectra, 20, output_folder = "./positive_mode/exact_matches_test_sets_splits/)

# Run for negative mode spectra
split_k_fold_cross_validation_analogue_test_set(negative_annotated_spectra, 20, output_folder = "./negative_mode/analogue_test_sets_splits/)
split_k_fold_cross_validation_exact_match_test_set(negative_annotated_spectra, 20, output_folder = "./negative_mode/exact_matches_test_sets_splits/)

# The 20 different datasplits will be stored in the specified folders
```

To train the models and to create the test results for MS2Query and all benchmarking methods (cosine, modified cosine and MS2Deepscore) run for each of the test split. So the script should be started 20 times for each type of test split. The running time for this script is a few days, since it trains all models and creates all test results. 

```python
from ms2query.benchmarking.k_fold_cross_validation import train_models_and_test_result_from_k_fold_folder
k_fold_split_number = 0 # Vary this number between 0 and 19
train_models_and_test_result_from_k_fold_folder(
    "./benchmarking_test_sets/exact_matches_test_sets_splits/",
    k_fold_split_number,
    exact_matches=True) # Change for analogue test set, this will change the precursor m/z prefiltering to match exact matches or analogue search for the reference benchmarking methods. 
```
After creating all the results, run the create_plot (the first block of python code) to create the new plots. 

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
