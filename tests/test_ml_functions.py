import os
import pandas as pd
from ms2query.utils import json_loader
from ms2query.s2v_functions import process_spectrums
from ms2query.ml_functions import find_basic_info


def test_find_basic_info():
    """Test find_basic_info"""
    path_tests = os.path.dirname(__file__)
    testfile_l = os.path.join(path_tests, "testspectrum_library.json")
    spectrums_l = json_loader(open(testfile_l))
    documents_l = process_spectrums(spectrums_l)
    test_matches_file = os.path.join(path_tests, "test_found_matches.csv")
    test_matches = pd.read_csv(test_matches_file, index_col=0)
    new_test_matches = find_basic_info(test_matches, documents_l)
    assert isinstance(new_test_matches, pd.DataFrame),\
        "Expected output to be df"
    assert "parent_mass" in new_test_matches.columns,\
        "Expected parent_mass to be added as a column"
