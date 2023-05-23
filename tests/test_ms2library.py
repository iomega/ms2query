import math
import os
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from ms2query.ms2library import (MS2Library,
                                 create_library_object_from_one_dir)
from ms2query.utils import load_pickled_file, SettingsRunMS2Query, column_names_for_output
from tests.test_utils import check_correct_results_csv_file


@pytest.fixture
def expected_ms2deespcore_scores():
    ms2dscores:pd.DataFrame = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_ms2ds_scores.pickle'))
    return ms2dscores


def test_get_all_ms2ds_scores(ms2library, test_spectra, expected_ms2deespcore_scores):
    """Test get_all_ms2ds_scores method of ms2library"""
    result = ms2library._get_all_ms2ds_scores(test_spectra[0])
    assert_series_equal(result, expected_ms2deespcore_scores)


def test_get_s2v_scores(ms2library, test_spectra):
    """Test _get_s2v_scores method of MS2Library"""
    result = ms2library._get_s2v_scores(
        test_spectra[0], [18, 68])
    assert np.allclose(result, np.array([0.97565603, 0.97848464])), \
        "Expected different Spec2Vec scores"


def test_get_average_ms2ds_for_inchikey14(ms2library):
    inchickey14s = {"BKUKTJSDOUXYFL", "BTVYFIMKUHNOBZ"}
    ms2ds_scores = pd.Series(
        [0.1, 0.8, 0.3],
        index=[87, 71, 73])
    results = ms2library._get_average_ms2ds_for_inchikey14(
        ms2ds_scores, inchickey14s)
    assert results == \
           {'BKUKTJSDOUXYFL': 0.1, 'BTVYFIMKUHNOBZ': 0.55}, \
           "Expected different results"


def test_get_chemical_neighbourhood_scores(ms2library):
    average_inchickey_scores = \
        {'BKUKTJSDOUXYFL': 0.8,
         'UZMVEOVJASEKLP': 0.8,
         'QWSYKJZSJYRUSS': 0.8,
         'GRVRRAOIXXYICO': 0.8,
         'WXDBUBIFYCCNLE': 0.8,
         'ORRFIXSGNXBETO': 0.7,
         'LLWMPGSQZXZZAE': 0.7,
         'CTBBEXWJRAPJIZ': 0.6,
         'YQLQWGVOWKPLFR': 0.6,
         'BTVYFIMKUHNOBZ': 0.6}

    results = ms2library._calculate_average_multiple_library_structures({"BKUKTJSDOUXYFL"}, average_inchickey_scores)
    assert isinstance(results, dict), "expected a dictionary"
    assert len(results) == 1, "Expected different number of results in " \
                              "dictionary"
    assert 'BKUKTJSDOUXYFL' in results
    scores = results['BKUKTJSDOUXYFL']
    assert isinstance(scores, tuple)
    assert len(scores) == 2, "Expected two scores for each InChiKey"
    assert math.isclose(scores[0], 0.72)
    assert math.isclose(scores[1], 0.4607757103045708)


def test_analog_search_store_in_csv(ms2library, test_spectra, tmp_path):
    results_csv_file = os.path.join(tmp_path, "test_csv_analog_search")
    settings = SettingsRunMS2Query(additional_metadata_columns=(("spectrum_id", )))
    ms2library.analog_search_store_in_csv(test_spectra, results_csv_file, settings)
    assert os.path.exists(results_csv_file)
    expected_headers = \
        ['query_spectrum_nr', "ms2query_model_prediction", "precursor_mz_difference", "precursor_mz_query_spectrum",
         "precursor_mz_analog", "inchikey", "analog_compound_name", "smiles", "spectrum_id"]
    check_correct_results_csv_file(
        pd.read_csv(results_csv_file),
        expected_headers)


def test_create_library_object_from_one_dir():
    """Test creating a MS2Library object with create_library_object_from_one_dir"""
    path_to_tests_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files')
    library = create_library_object_from_one_dir(path_to_tests_dir)
    assert isinstance(library, MS2Library)


def test_analog_yield_df(ms2library, test_spectra, tmp_path):
    settings = SettingsRunMS2Query(additional_metadata_columns=("spectrum_id", ),)
    result = ms2library.analog_search_yield_df(test_spectra, settings)
    expected_headers = \
        ['query_spectrum_nr', "ms2query_model_prediction", "precursor_mz_difference", "precursor_mz_query_spectrum",
         "precursor_mz_analog", "inchikey", "analog_compound_name", "smiles", "spectrum_id"]
    check_correct_results_csv_file(list(result)[0], expected_headers, nr_of_rows_to_check=1)


def test_analog_yield_df_additional_columns(ms2library, test_spectra, tmp_path):
    settings = SettingsRunMS2Query(additional_metadata_columns=("charge", ),
                                   additional_ms2query_score_columns=("s2v_score", "ms2ds_score",),)
    result = ms2library.analog_search_yield_df(test_spectra, settings)
    result_first_spectrum = list(result)[0]
    check_correct_results_csv_file(result_first_spectrum,
                                   column_names_for_output(True, True, ("charge",),
                                                           ("s2v_score", "ms2ds_score",)),
                                   nr_of_rows_to_check=1)

