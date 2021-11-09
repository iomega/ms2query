import os
from typing import Union, Dict
from tqdm import tqdm
from ms2query.ms2library import MS2Library
from ms2query.utils import add_unknown_charges_to_spectra
from ms2query.utils import convert_files_to_matchms_spectrum_objects
from urllib.request import urlretrieve


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
                        results_folder: str,
                        nr_of_analogs_to_store: int = 1,
                        minimal_ms2query_score: Union[int, float] = 0.7,
                        analog_search: bool = True,
                        library_search: bool = True
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
        In this folder new folders are created called "analog_search" and "library_search"
        containing the csv files with results.
    nr_of_top_analogs_to_store:
        The number of returned analogs that are stored.
    minimal_ms2query_score:
        The minimal ms2query metascore needed to be stored in the csv file.
        Spectra for which no analog with this minimal metascore was found,
        will not be stored in the csv file.
    analog_search:
        If True an analog search is performed and the results are stored
    library_search:
        If True a library search (Finding true matches) is performed
    """
    # pylint: disable=too-many-arguments

    # check if there is a results folder otherwise create one
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    if analog_search:
        if not os.path.exists(os.path.join(results_folder, "analog_search")):
            os.mkdir(os.path.join(results_folder, "analog_search"))
    if library_search:
        if not os.path.exists(os.path.join(results_folder, "library_search")):
            os.mkdir(os.path.join(results_folder, "library_search"))

    # Go through spectra files in directory
    for file_name in os.listdir(folder_with_spectra):
        file_path = os.path.join(folder_with_spectra, file_name)
        # skip folders
        if os.path.isfile(file_path):
            spectra = convert_files_to_matchms_spectrum_objects(os.path.join(folder_with_spectra, file_name))
            if spectra is not None:
                add_unknown_charges_to_spectra(spectra)
                if analog_search:
                    analogs_results_file_name = os.path.join(results_folder, "analog_search", os.path.splitext(file_name)[0] + ".csv")
                    ms2library.analog_search_store_in_csv(spectra,
                                                          analogs_results_file_name,
                                                          nr_of_top_analogs_to_save=nr_of_analogs_to_store,
                                                          minimal_ms2query_metascore=minimal_ms2query_score)
                if library_search:
                    library_results_file_name = os.path.join(results_folder, "library_search", os.path.splitext(file_name)[0] + ".csv")
                    ms2library.store_potential_true_matches(spectra,
                                                            library_results_file_name)
