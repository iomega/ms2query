import json
import os
from typing import Union
from urllib.request import urlopen, urlretrieve
from ms2query.ms2library import MS2Library
from ms2query.utils import load_matchms_spectrum_objects_from_file, SettingsRunMS2Query, return_non_existing_file_name


def download_zenodo_files(zenodo_doi: int,
                          dir_to_store_files:str):
    """Downloads files from Zenodo

    Args:
    ------
    zenodo_doi: 
        The doi of the zenodo files you would like to download
    dir_to_store_files:
        The path to the directory in which the downloaded files will be stored.
        The directory does not have to exist yet.
        """
    if not os.path.exists(dir_to_store_files):
        os.mkdir(dir_to_store_files)
    file_names_metadata_url = "https://zenodo.org/api/records/" + str(zenodo_doi)
    with urlopen(file_names_metadata_url) as zenodo_metadata_file:
        file_names_metadata_json: dict = json.loads(zenodo_metadata_file.read())
    files = file_names_metadata_json["files"]
    zenodo_files_url = f"https://zenodo.org/record/{zenodo_doi}/files/"

    for file in files:
        file_name = file["key"]
        store_file_location = os.path.join(dir_to_store_files, file_name)
        if not os.path.exists(store_file_location):
            print(f"downloading the file {file_name} from zenodo ({file['size'] / 1000000:.1f} MB)")
            urlretrieve(zenodo_files_url + file_name,
                        store_file_location)
        else:
            print(f"file with the name {store_file_location} already exists, so was not downloaded")


def run_complete_folder(ms2library: MS2Library,
                        folder_with_spectra: str,
                        results_folder: Union[str, None] = None,
                        settings: SettingsRunMS2Query = None
                        ) -> None:
    """Stores analog and library search results for all spectra files in folder

    Args:
    ------
    ms2library:
        MS2Library object
    folder_with_spectra:
        Path to folder containing spectra on which analog search should be run.
    results_folder:
        Path to folder in which the results are stored, folder does not have to exist yet.
        In this folder the csv files with results are stored. When None results_folder is set to
        folder_with_spectra/result.
    settings:
        Settings for running MS2Query, see SettingsRunMS2Query for details.
    """
    folder_contained_spectrum_file = False

    # Go through spectra files in directory
    for file_name in os.listdir(folder_with_spectra):
        file_path = os.path.join(folder_with_spectra, file_name)
        # skip folders
        if os.path.isfile(file_path):
            if os.path.splitext(file_name)[1].lower() in {".mzml", ".json", ".mgf", ".msp", ".mzxml", ".usi", ".pickle"}:
                run_ms2query_single_file(spectrum_file_name=file_name,
                                         folder_with_spectra=folder_with_spectra,
                                         results_folder=results_folder,
                                         ms2library=ms2library, settings=settings)
                folder_contained_spectrum_file = True
            else:
                print(f'The file extension of the file {file_name} is not recognized, this file was therefore skipped, '
                      f'accepted file types are ".mzml", ".json", ".mgf", ".msp", ".mzxml", ".usi" or ".pickle"')
    if folder_contained_spectrum_file is False:
        print(f"The specified spectra folder does not contain any file with spectra. "
              f"The folder given is {folder_with_spectra}")


def run_ms2query_single_file(spectrum_file_name,
                             folder_with_spectra,
                             results_folder,
                             ms2library,
                             settings):
    """Runs MS2Query on a single file

    Args:
    ------
    spectrum_file_name:
        The file name of a file contain mass spectra, accepted file types are
        ".mzML", ".json", ".mgf", ".msp", ".mzxml", ".usi" or ".pickle"
    folder_with_spectra:
        Path to folder containing spectra on which analog search should be run.
    results_folder:
        Path to folder in which the results are stored, folder does not have to exist yet.
        In this folder the csv files with results are stored. When None results_folder is set to
        folder_with_spectra/result.
    ms2library:
        MS2Library object
    settings:
        Settings for running MS2Query, see SettingsRunMS2Query for details.
    """
    if results_folder is None:
        results_folder = os.path.join(folder_with_spectra, "results")
    # check if there is a results folder otherwise create one
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    spectra = load_matchms_spectrum_objects_from_file(os.path.join(folder_with_spectra, spectrum_file_name))
    analogs_results_file_name = return_non_existing_file_name(
        os.path.join(results_folder,
                     os.path.splitext(spectrum_file_name)[0] + ".csv"))
    ms2library.analog_search_store_in_csv(spectra,
                                          analogs_results_file_name,
                                          settings)
    print(f"Results stored in {analogs_results_file_name}")
