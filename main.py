import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ms2query.run_ms2query import download_zenodo_files, run_complete_folder, run_ms2query_single_file
from ms2query.ms2library import create_library_object_from_one_dir


def is_directory(path: str):
    assert os.path.exists(path), f"The specified path does not exist"
    return path

# todo
#  Check if the ionization mode of the library is positive
#  Make automatic splitting spectra in positive and negative mode possible
# todo add option for verbosity.
#  Make sure tensorflow does not show any warnings
# todo store the settings of the model, like positive mode and nr of training spectra that should be used.


def command_line():
    parser = argparse.ArgumentParser(
        prog='MS2Query',
        description='MS2Query is a tool for MSMS library matching, searching both for analogues and exact matches in one run')
    parser.add_argument('spectra', type=is_directory, help='The MS2 query spectra that should be processed. '
                                                           'If a directory is specified all spectrum files in the directory will be processed. '
                                                           'Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object')
    parser.add_argument('library', type=is_directory, help="The directory containing the library spectra, to download add --download")
    parser.add_argument("--download",
                        action="store_true",
                        help= "This will download the most up to date model and library in the ionization mode specified under --mode")
    parser.add_argument("--mode", action="store", choices=["positive", "negative"], help="Specify the ionization mode of the spectra")
    parser.add_argument("--results", action="store", help="The folder in which the results should be stored, the default is a new results folder in the folder with the spectra")

    args = parser.parse_args()
    ms2query_library_files_directory = args.library
    ms2_spectra_location = args.spectra
    ion_mode = args.mode
    results_folder = args.results
    if args.download:
        zenodo_DOIs = {"positive": 6997924,
                       "negative": 7107654}
        download_zenodo_files(zenodo_DOIs[ion_mode],
                              ms2query_library_files_directory)

    # Create a MS2Library object
    ms2library = create_library_object_from_one_dir(ms2query_library_files_directory)

    if os.path.isfile(ms2_spectra_location):
        folder_with_spectra, spectrum_file_name = os.path.split(ms2_spectra_location)

        run_ms2query_single_file(spectrum_file_name=spectrum_file_name,
                                 folder_with_spectra=folder_with_spectra,
                                 results_folder=results_folder,
                                 ms2library=ms2library,
                                 settings=None)
    else:
        # Run library search and analog search on your files.
        run_complete_folder(ms2library, ms2_spectra_location, results_folder=results_folder)
