from ms2query.ms2library import create_library_object_from_one_dir
from ms2query.run_ms2query import (default_library_file_base_names,
                                   download_default_models,
                                   run_complete_folder)


# Set the location where all your downloaded model files are stored
ms2query_library_files_directory = "../data/models_embeddings_files/"
# define the folder in which your query spectra are stored.
# Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object.
ms2_spectra_directory = "../data/test_dir/test_spectra/"

# Downloads pretrained models and files for MS2Query (>10GB download)
download_default_models(ms2query_library_files_directory, default_library_file_base_names())

# Create a MS2Library object
ms2library = create_library_object_from_one_dir(ms2query_library_files_directory, default_library_file_base_names())

# Run library search and analog search on your files.
run_complete_folder(ms2library, ms2_spectra_directory)
