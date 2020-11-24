from ms2query.app_helpers import gather_test_json
from ms2query.app_helpers import gather_zenodo_library
from ms2query.app_helpers import get_zenodo_models_dict


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


def test_gather_zenodo_library():
    """Test gather_zenodo_library"""
    lib_dict = gather_zenodo_library("downloads")
    lib_dict_keys = list(lib_dict.keys())
    first_record = lib_dict[lib_dict_keys[0]]
    assert isinstance(lib_dict, dict), "Expected output to be dict"
    assert all(isinstance(lib_dict_key, str)
               for lib_dict_key in lib_dict_keys), "Expected keys to be str"
    assert isinstance(first_record, tuple), "Expected output to be tuple"
    assert all(isinstance(first_record[i], str) for i in range(2)),\
        "Expected first two elements to be str"
    assert isinstance(first_record[2], int), "Expected third element to be int"


def test_get_zenodo_models_dict():
    """Test get_zenodo_models_dict"""
    model_dict = get_zenodo_models_dict("downloads")
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
