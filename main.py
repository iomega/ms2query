import os
import argparse
import pathlib
# from ms2query.run_ms2query import download_zenodo_files, run_complete_folder
# from ms2query.ms2library import create_library_object_from_one_dir
# os.environ["TF CPP MIN LOG LEVEL"] = "3"


def is_directory(path: str):
    assert os.path.isdir(path), f"The specified path is not a directory"
    return path

# todo Add option to just specify one file,
#  dont raise an error when file already exists,
#  maybe a function for overwriting a file
#  Check if the ionization mode of the library is positive
#  Make automatic splitting spectra in positive and negative mode possible


def command_line():
    parser = argparse.ArgumentParser(
        prog='MS2Query',
        description='MS2Query is a tool for MSMS library matching, searching both for analogues and exact matches in one run')
    parser.add_argument('spectra', type=is_directory, help="The directory containing the query spectra")
    parser.add_argument('library', type=is_directory, help="The directory containing the library spectra, to download add --download")
    parser.add_argument("--download",
                        action="store_true", help="This will download the most up to date model and library in the ionization mode specified under --mode")
    parser.add_argument("--mode", action="store", choices=["positive", "negative"], help = "Specify the ionization mode of the spectra")
    # add optional parser for results folder location, default is results

    args = parser.parse_args()
    print(args.spectra)
    print(args)
    print(args.download)


    # # Set the location where downloaded library and model files are stored
    # ms2query_library_files_directory = "./ms2query_library_files"
    #
    # # Define the folder in which your query spectra are stored.
    # # Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object.
    # ms2_spectra_directory =
    # ion_mode =  # Fill in "positive" or "negative" to indicate for which ion mode you would like to download the library
    #
    # zenodo_DOIs = {"positive": 6997924,
    #                "negative": 7107654}
    #
    # # Downloads pretrained models and files for MS2Query (>2GB download)
    # download_zenodo_files(zenodo_DOIs[ion_mode],
    #                       ms2query_library_files_directory)
    #
    # # Create a MS2Library object
    # ms2library = create_library_object_from_one_dir(ms2query_library_files_directory)
    #
    # # Run library search and analog search on your files.
    # run_complete_folder(ms2library, ms2_spectra_directory)