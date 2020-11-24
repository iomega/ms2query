import os
import numpy as np
import pandas as pd
from spec2vec import SpectrumDocument
from ms2query.app_helpers import gather_test_json
from ms2query.app_helpers import get_query
from ms2query.app_helpers import make_downloads_folder
from ms2query.app_helpers import get_library_data
from ms2query.app_helpers import gather_zenodo_library
from ms2query.app_helpers import get_model
from ms2query.app_helpers import get_zenodo_models
from ms2query.app_helpers import get_zenodo_models_dict
from ms2query.app_helpers import url_to_file
from ms2query.app_helpers import do_spectrum_processing
from ms2query.app_helpers import get_example_library_matches
from ms2query.app_helpers import get_library_similarities
from ms2query.utils import json_loader
from ms2query.s2v_functions import process_spectrums


# only functions are tested that do not depend on any streamlit commands

def test_gather_test_json():
    """Test gather_test_json"""
    test_name = "testspectrum_library.json"
    testlib_dict, testlib_list = gather_test_json(test_name)
    out_path = testlib_dict[test_name]
    assert isinstance(testlib_dict, dict), "Expected output to be dict"
    assert isinstance(testlib_list, list), "Expected output to be list"
    assert test_name in testlib_dict, "Expected test_name in testlib_dict"
    assert test_name in testlib_list, "Expected test_name in testlib_list"
    assert isinstance(out_path, str), "Expected output to be str"


def test_get_query():
    """Test get_query"""
    res = get_query()
    assert isinstance(res, list), "Expected output to be list"
    assert len(res) == 0, "Expected default output from get_query to be empty"


def test_make_downloads_folder():
    """Test make_downloads_folder"""
    downloads_folder = make_downloads_folder()
    assert downloads_folder.endswith("downloads"), \
        "Expected default output to end in 'downloads'"
    assert isinstance(downloads_folder, str)


def test_get_library_data():
    """Test get_library_data"""
    res = get_library_data("downloads")
    assert isinstance(res, tuple), "Expected output to be tuple"
    assert isinstance(res[0], list), "Expected output to be list"
    assert len(res[0]) == 0, "Expected default output to be empty"
    assert isinstance(res[1], bool), "Expected output to be bool"
    assert res[1] is False, "Expected default output to be False"
    assert res[2] is None, "Expected default output to be None"


def test_gather_zenodo_library():
    """Test gather_zenodo_library"""
    download = "downloads"
    lib_dict = gather_zenodo_library(download)
    lib_dict_keys = list(lib_dict.keys())
    first_record = lib_dict[lib_dict_keys[0]]
    assert isinstance(lib_dict, dict), "Expected output to be dict"
    assert all(isinstance(lib_dict_key, str)
               for lib_dict_key in lib_dict_keys), "Expected keys to be str"
    assert isinstance(first_record, tuple), "Expected output to be tuple"
    assert all(isinstance(first_record[i], str) for i in range(2)), \
        "Expected first two elements to be str"
    assert isinstance(first_record[2], int), "Expected third element to be int"
    assert download in first_record[1], \
        "Expected download to be added to file path"


def test_get_model():
    """Test get_model"""
    model, model_num = get_model("downloads")
    assert model is None, "Expected default output to be empty"
    assert model_num is None, "Expected default output to be empty"


def test_get_zenodo_models():
    """Test get_zenodo_models"""
    model_name, model_file, model_num = get_zenodo_models()
    assert isinstance(model_name, str), "Expected output to be str"
    assert len(model_name) == 0, "Expected default output to be empty"
    assert model_file is None, "Expected default output to be empty"
    assert model_num is None, "Expected default output to be empty"


def test_get_zenodo_models_dict():
    """Test get_zenodo_models_dict"""
    download = "downloads"
    model_dict = get_zenodo_models_dict(download)
    model_dict_keys = list(model_dict.keys())
    first_record = model_dict[model_dict_keys[0]]
    assert isinstance(model_dict, dict), "Expected output to be dict"
    assert all(isinstance(model_dict_key, str) for model_dict_key
               in model_dict_keys), "Expected keys to be str"
    assert isinstance(first_record, tuple), "Expected output to be tuple"
    assert all(isinstance(first_record[i], list) for i in range(2)), \
        "Expected first two elements to be list"
    assert all(isinstance(first_record_elem, str)
               for first_record_elem in first_record[0]), \
        "Expected first two elements to be str"
    assert isinstance(first_record[2], int), "Expected third element to be int"
    assert download in first_record[1][0], \
        "Expected download to be added to file path"


def test_url_to_file():
    """Test url_to_file"""
    download = "downloads"
    file_name = "some_file"
    mock_urls = [f"a_mock_url/{file_name}?download"]
    mock_files = url_to_file(mock_urls, download)
    mock_file = mock_files[0]
    assert isinstance(mock_files, list), "Expected output to be list"
    assert isinstance(mock_file, str), "Expected output to be str"
    assert mock_file.endswith(file_name), \
        "Expected mock_file to end with file_name if path is made correctly"
    assert mock_file.startswith(download), \
        "Expected mock_file to start with downloads if path is made correctly"


def test_do_spectrum_processing():
    """Test do_spectrum_processing"""
    path_tests = os.path.dirname(__file__)
    q_testfile = os.path.join(path_tests, "testspectrum_query.json")
    q_spectrums = json_loader(open(q_testfile))
    l_testfile = os.path.join(path_tests, "testspectrum_library.json")
    l_spectrums = json_loader(open(l_testfile))
    q_documents, l_documents = do_spectrum_processing(
        q_spectrums, l_spectrums, False)
    assert isinstance(q_documents, list), "Expected output to be list."
    assert isinstance(q_documents[0], SpectrumDocument), \
        "Expected output to be SpectrumDocument."
    assert q_documents[0]._obj.get("spectrum_id") == q_spectrums[0] \
        .metadata["spectrum_id"]
    assert isinstance(l_documents, list), "Expected output to be list."
    assert isinstance(l_documents[0], SpectrumDocument), \
        "Expected output to be SpectrumDocument."
    assert l_documents[0]._obj.get("spectrum_id") == l_spectrums[0] \
        .metadata["spectrum_id"]
    q_documents_2, l_documents_2 = do_spectrum_processing(
        q_spectrums, l_documents, True)
    assert all([isinstance(l_documents_2[0], SpectrumDocument),
                isinstance(l_documents[0], SpectrumDocument)]), \
        "Expected output to be the same type: SpectrumDocument"


def test_get_example_library_matches():
    """Test get_example_library_matches"""
    test_matches = get_example_library_matches()
    assert isinstance(test_matches, pd.DataFrame), "Expected output to be df"


def test_get_library_similarities():
    """Test get_library_similarities"""
    path_tests = os.path.dirname(__file__)
    testfile = os.path.join(path_tests, "testspectrum_library.json")
    spectrums = json_loader(open(testfile))
    documents = process_spectrums(spectrums)
    test_matches = get_example_library_matches()
    test_sim_matrix = get_library_similarities(test_matches, documents, 0)
    assert isinstance(test_sim_matrix, np.ndarray),\
        "Expected output to be ndarray"
    assert test_sim_matrix.shape[0] == 5, "Expected 5 rows"
