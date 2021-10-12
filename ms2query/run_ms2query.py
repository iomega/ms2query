import os
from typing import List, Union, Dict
from tqdm import tqdm
from matchms.importing.load_from_mzml import load_from_mzml
from matchms.Spectrum import Spectrum
from ms2query.ms2library import MS2Library
from ms2query.utils import load_pickled_file
from urllib.request import urlretrieve


def default_library_file_names() -> Dict[str, str]:
    """Returns a dictionary with the file names of default files for a MS2Library"""
    return {"sqlite": "ALL_GNPS_210409_train_split.sqlite",
            "classifiers": "ALL_GNPS_210409_positive_processed_annotated_CF_NPC_classes.txt",
            "s2v_model": "ALL_GNPS_210409_Spec2Vec_ms2query.model",
            "ms2ds_model": "ms2ds_20210420-141937_data210409_10k_500_500_200.hdf5",
            "ms2query_model": "ms2query_model_all_scores_dropout_regularization.hdf5",
            "s2v_embeddings": "s2v_embeddings_train_spectra_210426.pickle",
            "ms2ds_embeddings": "ms2ds_embeddings_train_spectra_210426.pickle"}


def create_default_library_object(directory: str,
                                  file_name_dictionary: Dict[str, str]
                                  ) -> MS2Library:
    """Creates a library object for specified directory and file names

    For default file names the function default_library_file_names can be run.

    Args:
    ------
    directory:
        Path to the directory in which the files are stored
    file_name_dictionary:
        A dictionary with as keys the type of file and as values the file names
    """
    sqlite_file = os.path.join(directory, file_name_dictionary["sqlite"])
    classifiers_file = os.path.join(directory, file_name_dictionary["classifiers"])

    # Models
    s2v_model_file = os.path.join(directory, file_name_dictionary["s2v_model"])
    ms2ds_model_file = os.path.join(directory, file_name_dictionary["ms2ds_model"])
    ms2query_model = os.path.join(directory, file_name_dictionary["ms2query_model"])

    # Embeddings
    s2v_embeddings_file = os.path.join(directory, file_name_dictionary["s2v_embeddings"])
    ms2ds_embeddings_file = os.path.join(directory, file_name_dictionary["ms2ds_embeddings"])

    return MS2Library(sqlite_file, s2v_model_file, ms2ds_model_file,
                      s2v_embeddings_file, ms2ds_embeddings_file,
                      ms2query_model, classifiers_file)


def automatically_download_models(dir_to_store_files: str,
                                  file_name_dict: Dict[str, str]
                                  ):
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
    zenodo_files_location = "https://zenodo.org/record/5564815/files/"
    for file_name in tqdm(file_name_dict.values(),
                          "Downloading library files"):
        complete_url = zenodo_files_location + file_name + "?download=1"
        file_location = dir_to_store_files + file_name
        if not os.path.exists(file_location):
            urlretrieve(complete_url, file_location)
        else:
            print(f"file with the name {file_location} already exists, so was not downloaded")


def run_analog_for_complete_folder(ms2library: MS2Library,
                                   folder_with_spectra,
                                   analog_results_folder
                                   ) -> None:
    """Stores a csv file with analog search results for all spectra in folder"""
    for file in os.listdir(folder_with_spectra):
        spectra = convert_files_to_matchms_files(file)
        ms2library.analog_search_store_in_csv(spectra, )
    # check if there is a results folder otherwise create one


def convert_files_to_matchms_files(file_location
                                   ) -> Union[List[Spectrum], None]:
    if file_location.endswith(".mzML"):
        spectra = list(load_from_mzml(file_location))
        return spectra
    #todo add options for loading other spectra. Also include pickled matchms files
    # todo add needed filtering steps? like set charge


if __name__ == "__main__":
    # spectra = load_pickled_file("../data/case_studies/Huali/correct_files_BN/positive/1d-BN-2.pickle")[:2]
    # library = create_default_library_object("../data/models_embeddings_files/", default_library_file_names())
    automatically_download_models("../data/test_dir/", {"classifiers": "ALL_GNPS_210409_positive_processed_annotated_CF_NPC_classes.txt"})
    # print(library)
    # library.analog_search_store_in_csv(spectra, "../tests/test_files/can_be_deleted.csv", 2)
    # classifiers_file = os.path.join("../data/models_embeddings_files/",
    #                                 "ALL_GNPS_210409_positive_processed_annotated_CF_NPC_classes.txt")
    # result_df = result[0].export_to_dataframe(2, classifiers_file)
