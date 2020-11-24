from ms2query.app_helpers import gather_test_json


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
