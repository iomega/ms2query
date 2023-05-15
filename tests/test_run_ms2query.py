import os
import sys

import pandas as pd

from ms2query.ms2library import create_library_object_from_one_dir, select_files_for_ms2query
from ms2query.run_ms2query import download_zenodo_files, run_complete_folder, zenodo_dois, available_zenodo_files
from ms2query.utils import SettingsRunMS2Query
from tests.test_ms2library import MS2Library
from tests.test_utils import check_correct_results_csv_file

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


def test_download_zenodo():
    """Tests if the files on zenodo match the load library from one dir settings"""
    for ionisation_mode in ["positive", "negative"]:
        zenodo_metadata_url, zenodo_files_url = zenodo_dois(ionisation_mode)
        file_names_and_sizes = available_zenodo_files(zenodo_metadata_url)
        file_names = [file_name for file_name in file_names_and_sizes]
        select_files_for_ms2query(file_names)


def test_download_models_only():
    """Tests if downloading the models only works"""
    for ionisation_mode in ["positive", "negative"]:
        zenodo_metadata_url, zenodo_files_url = zenodo_dois(ionisation_mode)
        file_names_and_sizes = available_zenodo_files(zenodo_metadata_url, only_models=True)
        file_names = [file_name for file_name in file_names_and_sizes]
        assert len(file_names) == 5


def test_download_default_models(tmp_path):
    """Tests downloading small files from zenodo

    The files are a total of 20 MB from https://zenodo.org/record/7108049#.Yy2nPKRBxPY"""
    run_test = False # Run test is set to false, since downloading takes too long for default testing
    if run_test:
        dir_to_store_positive_files = os.path.join(tmp_path, "positive_model")
        dir_to_store_negative_files = os.path.join(tmp_path, "negative_model")

        download_zenodo_files("positive", dir_to_store_positive_files)
        download_zenodo_files("negative", dir_to_store_negative_files)
        assert os.path.exists(dir_to_store_positive_files)
        assert os.path.exists(dir_to_store_negative_files)
        pos_ms2library = create_library_object_from_one_dir(dir_to_store_positive_files)
        neg_ms2library = create_library_object_from_one_dir(dir_to_store_negative_files)
        assert isinstance(pos_ms2library, MS2Library)
        assert isinstance(neg_ms2library, MS2Library)


def create_test_folder_with_spectra_files(path, spectra):
    """Creates a folder with two files containing two test spectra"""
    spectra_files_folder = os.path.join(path, "spectra_files_folder")
    os.mkdir(spectra_files_folder)

    pickle.dump(spectra, open(os.path.join(spectra_files_folder, "spectra_file_1.pickle"), "wb"))
    pickle.dump(spectra, open(os.path.join(spectra_files_folder, "spectra_file_2.pickle"), "wb"))
    return spectra_files_folder


def test_run_complete_folder(tmp_path, ms2library, test_spectra):
    folder_with_spectra = create_test_folder_with_spectra_files(tmp_path, test_spectra)
    results_directory = os.path.join(folder_with_spectra, "results")

    run_complete_folder(ms2library=ms2library,
                        folder_with_spectra=folder_with_spectra)
    assert os.path.exists(results_directory), "Expected results directory to be created"
    assert os.listdir(os.path.join(results_directory)).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()

    expected_headers = ['query_spectrum_nr', 'ms2query_model_prediction', 'precursor_mz_difference',
                        'precursor_mz_query_spectrum', 'precursor_mz_analog', 'inchikey',
                        'analog_compound_name', 'smiles', 'retention_time', 'retention_index']
    check_correct_results_csv_file(pd.read_csv(os.path.join(os.path.join(results_directory, 'spectra_file_1.csv'))),
                                   expected_headers)
    check_correct_results_csv_file(pd.read_csv(os.path.join(os.path.join(results_directory, 'spectra_file_2.csv'))),
                                   expected_headers)


def test_run_complete_folder_with_classifiers(tmp_path, ms2library, test_spectra):
    folder_with_spectra = create_test_folder_with_spectra_files(tmp_path, test_spectra)
    results_directory = os.path.join(folder_with_spectra, "results")
    settings = SettingsRunMS2Query(minimal_ms2query_metascore=0,
                                   additional_metadata_columns=("charge",),
                                   additional_ms2query_score_columns=("s2v_score", "ms2ds_score"))
    run_complete_folder(ms2library=ms2library,
                        folder_with_spectra=folder_with_spectra,
                        settings=settings
                        )
    assert os.path.exists(results_directory), "Expected results directory to be created"

    assert os.listdir(os.path.join(results_directory)).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()

    expected_headers = \
        ['query_spectrum_nr', "ms2query_model_prediction", "precursor_mz_difference", "precursor_mz_query_spectrum",
         "precursor_mz_analog", "inchikey", "analog_compound_name", "smiles", "charge", "s2v_score",
         "ms2ds_score", "cf_kingdom", "cf_superclass", "cf_class", "cf_subclass", "cf_direct_parent",
         "npc_class_results", "npc_superclass_results", "npc_pathway_results"]
    check_correct_results_csv_file(
        pd.read_csv(os.path.join(os.path.join(results_directory, 'spectra_file_1.csv'))),
        expected_headers)
    check_correct_results_csv_file(
        pd.read_csv(os.path.join(os.path.join(results_directory, 'spectra_file_2.csv'))),
        expected_headers)
