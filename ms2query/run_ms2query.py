import json
import os
from typing import List, Tuple, Union
from urllib.request import urlopen, urlretrieve
from ms2query.ms2library import MS2Library
from ms2query.utils import load_matchms_spectrum_objects_from_file


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
                        nr_of_analogs_to_store: int = 1,
                        minimal_ms2query_score: Union[int, float] = 0.0,
                        additional_metadata_columns: Tuple[str] = ("retention_time", "retention_index",),
                        additional_ms2query_score_columns: List[str] = None
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
    nr_of_top_analogs_to_store:
        The number of returned analogs that are stored.
    minimal_ms2query_score:
        The minimal ms2query metascore needed to be stored in the csv file.
        Spectra for which no analog with this minimal metascore was found,
        will not be stored in the csv file.
    additional_metadata_columns:
        Additional columns with query spectrum metadata that should be added. For instance "retention_time".
    additional_ms2query_score_columns:
        Additional columns with scores used for calculating the ms2query metascore
        Options are: "mass_similarity", "s2v_score", "ms2ds_score", "average_ms2ds_score_for_inchikey14",
        "nr_of_spectra_with_same_inchikey14*0.01", "chemical_neighbourhood_score",
        "average_tanimoto_score_for_chemical_neighbourhood_score",
        "nr_of_spectra_for_chemical_neighbourhood_score*0.01"
    set_charge_to:
        The charge of all spectra that have no charge is set to this value. This is important for precursor m/z
        calculations. It is important that for positive mode charge is set to 1 and at negative mode charge is set to -1
        for correct precursor m/z calculations.
    change_all_charges:
        If True the charge of all spectra is set to this value. If False only the spectra that do not have a specified
        charge will be changed.
    """
    # pylint: disable=too-many-arguments

    if results_folder is None:
        results_folder = os.path.join(folder_with_spectra, "results")
    # check if there is a results folder otherwise create one
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # Go through spectra files in directory
    for file_name in os.listdir(folder_with_spectra):
        file_path = os.path.join(folder_with_spectra, file_name)
        # skip folders
        if os.path.isfile(file_path):
            if os.path.splitext(file_name)[1] in {".mzML", ".json", ".mgf", ".msp", ".mzxml", ".usi", ".pickle"}:
                spectra = load_matchms_spectrum_objects_from_file(os.path.join(folder_with_spectra, file_name))
                analogs_results_file_name = os.path.join(results_folder, os.path.splitext(file_name)[0] + ".csv")
                ms2library.analog_search_store_in_csv(spectra,
                                                      analogs_results_file_name,
                                                      nr_of_top_analogs_to_save=nr_of_analogs_to_store,
                                                      minimal_ms2query_metascore=minimal_ms2query_score,
                                                      additional_metadata_columns=additional_metadata_columns,
                                                      additional_ms2query_score_columns=additional_ms2query_score_columns)
                print(f"Results stored in {analogs_results_file_name}")
            else:
                print(f'The file extension of the file {file_name} is not recognized, '
                      f'accepted file types are ".mzML", ".json", ".mgf", ".msp", ".mzxml", ".usi" or ".pickle"')