import os
import pandas as pd
from ms2query.utils import json_loader
from ms2query.s2v_functions import process_spectrums
from ms2query.ml_functions import find_basic_info
from ms2query.ml_functions import transform_num_matches
from ms2query.ml_functions import find_mass_similarity
from ms2query.ml_functions import find_info_matches


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


def test_transform_num_matches():
    """Test transform_num_matches"""
    path_tests = os.path.dirname(__file__)
    test_matches_file = os.path.join(path_tests, "test_found_matches.csv")
    test_matches = pd.read_csv(test_matches_file, index_col=0)
    new_test_matches = transform_num_matches(test_matches)
    assert isinstance(new_test_matches, pd.DataFrame),\
        "Expected output to be df"
    assert isinstance(new_test_matches["cosine_matches"].iloc[0], float),\
        "Expected cosine matches to contain floats now"
    assert isinstance(new_test_matches["mod_cosine_matches"].iloc[0], float), \
        "Expected mod cosine matches to contain floats now"


def test_find_mass_similarity():
    """Test find_mass_similarity"""
    path_tests = os.path.dirname(__file__)
    testfile_l = os.path.join(path_tests, "testspectrum_library.json")
    spectrums_l = json_loader(open(testfile_l))
    documents_l = process_spectrums(spectrums_l)
    test_matches_file = os.path.join(path_tests, "test_found_matches.csv")
    test_matches = pd.read_csv(test_matches_file, index_col=0)
    new_test_matches = find_mass_similarity(test_matches, documents_l, 100.)
    assert isinstance(new_test_matches, pd.DataFrame), \
        "Expected output to be df"
    assert "mass_sim" in new_test_matches.columns, \
        "Expected mass_sim to be added as a column"
    assert isinstance(new_test_matches["mass_sim"].iloc[0], float), \
        "Expected mass_sim to contain floats"


def test_find_info_matches():
    """Test find_info_matches"""
    path_tests = os.path.dirname(__file__)
    testfile_q = os.path.join(path_tests, "testspectrum_query.json")
    spectrums_q = json_loader(open(testfile_q))
    testfile_l = os.path.join(path_tests, "testspectrum_library.json")
    spectrums_l = json_loader(open(testfile_l))
    documents_q = process_spectrums(spectrums_q)
    documents_l = process_spectrums(spectrums_l)
    test_matches_file = os.path.join(path_tests, "test_found_matches.csv")
    test_matches = pd.read_csv(test_matches_file, index_col=0)
    new_test_matches = find_info_matches(
        [test_matches], documents_q, documents_l, add_mass_transform=True,
        max_parent_mass=1000.)
    assert isinstance(new_test_matches, list), \
        "Expected output to be list"
    assert isinstance(new_test_matches[0], pd.DataFrame), \
        "Expected output to be df"
    assert "mass_sim" in new_test_matches[0].columns, \
        "Expected mass_sim to be added as a column"
    assert isinstance(new_test_matches[0]["mass_sim"].iloc[0], float), \
        "Expected mass_sim to contain floats"
    assert isinstance(new_test_matches[0]["cosine_matches"].iloc[0], float), \
        "Expected cosine matches to contain floats now"
    assert isinstance(
        new_test_matches[0]["mod_cosine_matches"].iloc[0], float), \
        "Expected mod cosine matches to contain floats now"
    assert "parent_mass" in new_test_matches[0].columns, \
        "Expected parent_mass to be added as a column"
