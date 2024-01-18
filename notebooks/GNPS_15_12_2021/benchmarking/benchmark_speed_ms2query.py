"""This file was run to test the speed of MS2Query.

It took 4463.9 seconds to find matches for two files with 2987 and 3000 spectra on a Lenovo Thinkbook 15-IIL.
This laptop has an 11th generation Intel Core i5-1125G7 and 16GB installed RAM.
"""

import os
import time
from ms2query.ms2library import MS2Library
from ms2query.run_ms2query import run_complete_folder


start_time = time.time()

path_root = os.path.dirname(os.getcwd())
path_library = os.path.join(path_root, "../data/libraries_and_models/gnps_15_12_2021/library_gnps_15_12/")
ms2_spectra_directory = "C:/Users/jonge094/PycharmProjects/PhD_MS2Query/ms2query/data/libraries_and_models/gnps_15_12_2021/benchmarking/test_spectra"

# Create a MS2Library object
ms2library = MS2Library(sqlite_file_name=os.path.join(path_library, "library_GNPS_15_12_2021.sqlite"),
                        s2v_model_file_name=os.path.join(path_library, "spec2vec_model_GNPS_15_12_2021.model"),
                        ms2ds_model_file_name=os.path.join(path_library, "ms2ds_model_GNPS_15_12_2021.hdf5"),
                        s2v_embeddings_file_name=os.path.join(path_library,
                                                              "library_GNPS_15_12_2021_s2v_embeddings.pickle"),
                        ms2ds_embeddings_file_name=os.path.join(path_library,
                                                                "library_GNPS_15_12_2021_ms2ds_embeddings.pickle"),
                        ms2query_model_file_name=os.path.join(path_library, "ms2query_random_forest_model.pickle"))

# Run library search and analog search on your files.
run_complete_folder(ms2library, ms2_spectra_directory)
print(time.time() -start_time)
