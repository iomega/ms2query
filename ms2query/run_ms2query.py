import os
from typing import Dict, List, Tuple, Union
from urllib.request import urlretrieve
from tqdm import tqdm
from ms2query.ms2library import MS2Library
from ms2query.utils import convert_files_to_matchms_spectrum_objects


def default_library_file_base_names() -> Dict[str, str]:
    """Returns a dictionary with the base names of default files for a MS2Library"""
    return {"sqlite": "ALL_GNPS_210409_train_split.sqlite",
            "sqlite_trainables": "ALL_GNPS_210409_Spec2Vec_ms2query.model.trainables.syn1neg.npy",
            "sqlite_vectors": "ALL_GNPS_210409_Spec2Vec_ms2query.model.wv.vectors.npy",
            "classifiers": "ALL_GNPS_210409_positive_processed_annotated_CF_NPC_classes.txt",
            "s2v_model": "ALL_GNPS_210409_Spec2Vec_ms2query.model",
            "ms2ds_model": "ms2ds_20210420-141937_data210409_10k_500_500_200.hdf5",
            "ms2query_model": "ms2query_model_all_scores_dropout_regularization.hdf5",
            "s2v_embeddings": "s2v_embeddings_train_spectra_210426.pickle",
            "ms2ds_embeddings": "ms2ds_embeddings_train_spectra_210426.pickle"}


def download_default_models(dir_to_store_files: str,
                            file_name_dict: Dict[str, str]):
    """Downloads files from Zenodo

    Args:
    ------
    dir_to_store_files:
        The path to the directory in which the downloaded files will be stored.
        The directory does not have to exist yet.
    file_name_dict:
        A dictionary with as keys the type of file and as values the file names
    """
    if not os.path.exists(dir_to_store_files):
        os.mkdir(dir_to_store_files)
    zenodo_files_location = "https://zenodo.org/record/5645246/files/"
    for file_name in tqdm(file_name_dict.values(),
                          "Downloading library files"):
        complete_url = zenodo_files_location + file_name + "?download=1"
        file_location = os.path.join(dir_to_store_files, file_name)
        if not os.path.exists(file_location):
            urlretrieve(complete_url, file_location)
        else:
            print(f"file with the name {file_location} already exists, so was not downloaded")


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
            spectra = convert_files_to_matchms_spectrum_objects(os.path.join(folder_with_spectra, file_name))
            if spectra is not None:
                analogs_results_file_name = os.path.join(results_folder, os.path.splitext(file_name)[0] + ".csv")
                ms2library.analog_search_store_in_csv(spectra,
                                                      analogs_results_file_name,
                                                      nr_of_top_analogs_to_save=nr_of_analogs_to_store,
                                                      minimal_ms2query_metascore=minimal_ms2query_score,
                                                      additional_metadata_columns=additional_metadata_columns,
                                                      additional_ms2query_score_columns=additional_ms2query_score_columns)
                print(f"Results stored in {analogs_results_file_name}")
