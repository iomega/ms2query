import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# pylint: disable=wrong-import-position
import argparse
import logging
from .__version__ import __version__
from .ms2library import MS2Library, create_library_object_from_one_dir
from .results_table import ResultsTable
from .run_ms2query import (download_zenodo_files, run_complete_folder,
                           run_ms2query_single_file)
from .utils import SettingsRunMS2Query


logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__all__ = [
    "__version__",
    "MS2Library",
    "ResultsTable",
]


def path_exists(path: str):
    assert os.path.exists(path), f"The specified path does not exist, the path given is {path}"
    return path


def command_line():
    parser = argparse.ArgumentParser(
        prog='MS2Query',
        description='MS2Query is a tool for MSMS library matching, '
                    'searching both for analogues and exact matches in one run')
    parser.add_argument('--spectra', action="store", type=path_exists,
                        help='The MS2 query spectra that should be processed. '
                             'If a directory is specified all spectrum files in the directory will be processed. '
                             'Accepted formats are: "mzML", "json", "mgf", "msp", "mzxml", "usi" or a pickled matchms object')
    parser.add_argument('--library', action="store", required=True, metavar="LIBRARY_FOLDER",
                        help="The directory containing the library spectra (in sqlite), models and precalculated embeddings, to download add --download")
    parser.add_argument("--ionmode", action="store", choices=["positive", "negative"],
                        help="Specify the ionization mode used")
    parser.add_argument("--download",
                        action="store_true",
                        help="This will download the most up to date model and library."
                             "The model will be stored in the folder given as the second argument"
                             "The model will be downloaded in the in the ionization mode specified under --mode")
    parser.add_argument("--results", action="store",
                        help="The folder in which the results should be stored. "
                             "The default is a new results folder in the folder with the spectra")
    parser.add_argument("--filter_ionmode", action="store_true",
                        help="Filter out all spectra that are not in the specified ion-mode. "
                             "The ion mode can be specified by using --ionmode")
    parser.add_argument("--additional_metadata", action="store",
                        default=("retention_time", "feature_id",),
                        nargs="+",
                        type=str,
                        help="Return additional metadata columns in the results, for example --additional_metadata retention_time feature_id")
    args = parser.parse_args()
    ms2query_library_files_directory = args.library
    ms2_spectra_location = args.spectra
    ion_mode = args.ionmode
    results_folder = args.results
    additional_columns = tuple(args.additional_metadata)
    if args.filter_ionmode:
        filter_ionmode = ion_mode
    else:
        filter_ionmode = None

    settings = SettingsRunMS2Query(filter_on_ion_mode=filter_ionmode,
                                   additional_metadata_columns=additional_columns)
    if args.download:
        assert ion_mode is not None, "Ion mode should be specified by adding --ion_mode"
        download_zenodo_files(ion_mode,
                              ms2query_library_files_directory)

    if ms2_spectra_location is not None:
        # Create a MS2Library object
        ms2library = create_library_object_from_one_dir(ms2query_library_files_directory)
        assert ms2library.ionization_mode == ion_mode, \
            f"The library used is in {ms2library.ionization_mode} ionization mode, while {ion_mode} is specified in --ionmode"

        if os.path.isfile(ms2_spectra_location):
            folder_with_spectra, spectrum_file_name = os.path.split(ms2_spectra_location)

            run_ms2query_single_file(spectrum_file_name=spectrum_file_name,
                                     folder_with_spectra=folder_with_spectra,
                                     results_folder=results_folder,
                                     ms2library=ms2library,
                                     settings=settings)
        else:
            # Run library search and analog search on your files.
            run_complete_folder(ms2library, ms2_spectra_location, results_folder=results_folder, settings=settings)
    if ms2_spectra_location is None and args.download is False:
        print("Nothing was run, please add --spectra or --downloads.")
