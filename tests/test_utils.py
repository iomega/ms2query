import os
from ms2query.utils import json_loader
from ms2query.utils import csv2dict


def test_json_loader_testfile():
    """Test creating spectrum(s) from test json file."""
    path_tests = os.path.dirname(__file__)
    testfile = os.path.join(path_tests, "testspectrum_query.json")
    spectrums = json_loader(open(testfile))
    assert isinstance(spectrums, list), "Expected output to be list."
    assert spectrums[0].metadata["spectrum_id"] == "CCMSLIB00000001655"
    

def test_csv2dict():
    """Test csv2dict"""
    base_name = os.path.split(os.path.dirname(__file__))[0]
    file_name = os.path.join(base_name, "model", "model_info.csv")
    csv_dict = csv2dict(file_name)
    assert isinstance(csv_dict, dict), "Expected output to be dict"
    assert "max_parent_mass" in csv_dict,\
        "Expected max_parent_mass to be in dict as it is in the file"
    assert isinstance(csv_dict["max_parent_mass"], list),\
        "Expected output to be list"
    assert isinstance(csv_dict["max_parent_mass"][0], str), \
        "Expected output to be str"
