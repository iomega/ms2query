import pandas as pd
from ms2query.app_helpers import gather_test_json
from ms2query.app_helpers import get_query
from ms2query.app_helpers import make_downloads_folder
from ms2query.app_helpers import get_library_data
from ms2query.app_helpers import gather_zenodo_library
from ms2query.app_helpers import get_zenodo_models_dict
from ms2query.app_helpers import url_to_file
from ms2query.app_helpers import get_example_library_matches


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
    assert downloads_folder.endswith("downloads"),\
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
    assert all(isinstance(first_record[i], str) for i in range(2)),\
        "Expected first two elements to be str"
    assert isinstance(first_record[2], int), "Expected third element to be int"
    assert download in first_record[1], \
        "Expected download to be added to file path"


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
    assert all(isinstance(first_record[i], list) for i in range(2)),\
        "Expected first two elements to be list"
    assert all(isinstance(first_record_elem, str)
               for first_record_elem in first_record[0]), \
        "Expected first two elements to be str"
    assert isinstance(first_record[2], int), "Expected third element to be int"
    assert download in first_record[1][0],\
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
    assert mock_file.endswith(file_name),\
        "Expected mock_file to end with file_name if path is made correctly"
    assert mock_file.startswith(download),\
        "Expected mock_file to start with downloads if path is made correctly"


def test_get_example_library_matches():
    """Test get_example_library_matches"""
    test_matches = get_example_library_matches()
    assert isinstance(test_matches, pd.DataFrame), "Expected output to be df"