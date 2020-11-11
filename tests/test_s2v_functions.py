import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from matchms.Spectrum import Spectrum
from ms2query.s2v_functions import set_spec2vec_defaults
from ms2query.s2v_functions import post_process_s2v
from ms2query.s2v_functions import process_spectrums
from ms2query.s2v_functions import library_matching
from ms2query.s2v_functions import get_metadata
from ms2query.s2v_functions import search_topn_s2v_matches
from ms2query.s2v_functions import search_parent_mass_matches
from ms2query.utils import json_loader
from spec2vec import SpectrumDocument


def test_post_process_s2v():
    """Test processing an individual spectrum with post_process_s2v"""
    path_tests = os.path.dirname(__file__)
    testfile = os.path.join(path_tests, "testspectrum_query.json")
    spectrums = json_loader(open(testfile))
    spectrum = post_process_s2v(spectrums[0])
    assert isinstance(spectrum, Spectrum), "Expected output to be Spectrum."
    assert spectrums[0].metadata["spectrum_id"] == spectrum\
        .metadata["spectrum_id"]


def test_set_spec2vec_defaults():
    """Test set_spec2vec_defaults"""
    x = 500
    settings = set_spec2vec_defaults(**{"mz_to": x})
    assert isinstance(settings, dict), "Expected output is dict"
    assert settings["mz_to"] == x


def test_process_spectrums():
    """Test process_spectrums"""
    path_tests = os.path.dirname(__file__)
    testfile = os.path.join(path_tests, "testspectrum_query.json")
    spectrums = json_loader(open(testfile))
    documents = process_spectrums(spectrums)
    assert isinstance(documents, list), "Expected output to be list."
    assert isinstance(documents[0], SpectrumDocument),\
        "Expected output to be SpectrumDocument."
    assert documents[0]._obj.get("spectrum_id") == spectrums[0]\
        .metadata["spectrum_id"]


def test_library_matching():
    """Test library_matching"""
    path_tests = os.path.dirname(__file__)
    testfile_q = os.path.join(path_tests, "testspectrum_query.json")
    spectrums_q = json_loader(open(testfile_q))
    testfile_l = os.path.join(path_tests, "testspectrum_library.json")
    spectrums_l = json_loader(open(testfile_l))
    documents_q = process_spectrums(spectrums_q)
    documents_l = process_spectrums(spectrums_l)
    test_model_file = os.path.join(path_tests,
                                   "testspectrum_library_model.model")
    test_model = Word2Vec.load(test_model_file)
    lib_length = len(documents_l)
    test_found_matches = library_matching(documents_q, documents_l, test_model,
                                          allowed_missing_percentage=100,
                                          presearch_based_on=[
                                              f"spec2vec-top{lib_length}"])
    assert isinstance(test_found_matches, list), "Expected output to be list"
    assert len(test_found_matches) == len(documents_q),\
        "Expected output for amount of query"
    assert isinstance(test_found_matches[0], pd.DataFrame),\
        "Expected output to be DataFrame"
    assert test_found_matches[0].shape[0] == lib_length,\
        "Expected number of matches to be number of library documents"


def test_get_metadata():
    """Test get_metadata"""
    path_tests = os.path.dirname(__file__)
    testfile_l = os.path.join(path_tests, "testspectrum_library.json")
    spectrums_l = json_loader(open(testfile_l))
    documents_l = process_spectrums(spectrums_l)
    smiles = get_metadata(documents_l)
    assert isinstance(smiles, list), "Expected a list"
    assert isinstance(smiles[0], str), "Expected smiles to be strings"
    assert len(smiles) == len(documents_l),\
        "Expected all docs to be turned into smiles"


def test_search_topn_s2v_matches():
    """Test search_topn_s2v_matches"""
    path_tests = os.path.dirname(__file__)
    testfile_q = os.path.join(path_tests, "testspectrum_query.json")
    spectrums_q = json_loader(open(testfile_q))
    testfile_l = os.path.join(path_tests, "testspectrum_library.json")
    spectrums_l = json_loader(open(testfile_l))
    documents_q = process_spectrums(spectrums_q)
    documents_l = process_spectrums(spectrums_l)
    test_model_file = os.path.join(path_tests,
                                   "testspectrum_library_model.model")
    test_model = Word2Vec.load(test_model_file)
    lib_length = len(documents_l)
    topn_s2v_matches, test_spec2vec_similarities = search_topn_s2v_matches(
                                               documents_q, documents_l,
                                               test_model,
                                               np.asarray(range(lib_length)),
                                               allowed_missing_percentage=100,
                                               presearch_based_on=[
                                                   f"spec2vec-top{lib_length}"]
                                               )
    assert isinstance(topn_s2v_matches, np.ndarray),\
        "Expected output to be ndarray"
    assert isinstance(test_spec2vec_similarities, np.ndarray), \
        "Expected output to be ndarray"
    assert topn_s2v_matches.shape == (lib_length, len(documents_q)),\
        "Expected shape to be (len(library), len(queries))"
    assert test_spec2vec_similarities.shape == (lib_length, len(documents_q)),\
        "Expected shape to be (len(library), len(queries))"


def test_search_parent_mass_matches():
    """Test search_parent_mass_matches"""
    path_tests = os.path.dirname(__file__)
    testfile_q = os.path.join(path_tests, "testspectrum_query.json")
    spectrums_q = json_loader(open(testfile_q))
    testfile_l = os.path.join(path_tests, "testspectrum_library.json")
    spectrums_l = json_loader(open(testfile_l))
    documents_q = process_spectrums(spectrums_q)
    documents_l = process_spectrums(spectrums_l)
    lib_length = len(documents_l)
    selection_massmatch, m_mass_matches = search_parent_mass_matches(
                                               documents_q, documents_l,
                                               np.asarray(range(lib_length)),
                                               presearch_based_on=[
                                                   "parentmass"])
    assert isinstance(selection_massmatch, list),\
        "Expected output to be list"
    assert isinstance(m_mass_matches, np.ndarray), \
        "Expected output to be ndarray"
    assert len(selection_massmatch) == len(documents_q),\
        "Expected len to be len(queries)"
    assert m_mass_matches.shape == (lib_length, len(documents_q)),\
        "Expected shape to be (len(library), len(queries))"
