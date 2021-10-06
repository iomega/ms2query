import os
import math
import pickle

import numpy as np
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from matchms import Spectrum
from tensorflow.keras.models import load_model as load_nn_model
from ms2query.ms2library import MS2Library
from ms2query.ms2library import get_ms2query_model_prediction_single_spectrum
from ms2query.utils import load_pickled_file
from ms2query.results_table import ResultsTable


@pytest.fixture
def file_names():
    """Returns file names of the files needed to create MS2Library object"""
    path_to_tests_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/')
    sqlite_file_loc = os.path.join(
        path_to_tests_dir,
        "general_test_files/100_test_spectra.sqlite")
    spec2vec_model_file_loc = os.path.join(
        path_to_tests_dir,
        "general_test_files/100_test_spectra_s2v_model.model")
    s2v_pickled_embeddings_file = os.path.join(
        path_to_tests_dir,
        "general_test_files/100_test_spectra_s2v_embeddings.pickle")
    ms2ds_model_file_name = os.path.join(
        path_to_tests_dir,
        "general_test_files/ms2ds_siamese_210301_5000_500_400.hdf5")
    ms2ds_embeddings_file_name = os.path.join(
        path_to_tests_dir,
        "general_test_files/100_test_spectra_ms2ds_embeddings.pickle")
    spectrum_id_column_name = "spectrumid"
    return sqlite_file_loc, spec2vec_model_file_loc, \
        s2v_pickled_embeddings_file, ms2ds_model_file_name, \
        ms2ds_embeddings_file_name, spectrum_id_column_name


@pytest.fixture
def test_spectra():
    """Returns a list with two spectra

    The spectra are created by using peaks from the first two spectra in
    100_test_spectra.pickle, to make sure that the peaks occur in the s2v
    model. The other values are random.
    """
    spectrum1 = Spectrum(mz=np.array([808.27356, 872.289917, 890.246277,
                                      891.272888, 894.326416, 904.195679,
                                      905.224548, 908.183472, 922.178101,
                                      923.155762], dtype="float"),
                         intensities=np.array([0.11106008, 0.12347332,
                                               0.16352988, 0.17101522,
                                               0.17312992, 0.19262333,
                                               0.21442898, 0.42173288,
                                               0.51071955, 1.],
                                              dtype="float"),
                         metadata={'pepmass': (907.0, None),
                                   'spectrumid': 'CCMSLIB00000001760',
                                   'precursor_mz': 907.0,
                                   'parent_mass': 905.9927235480093,
                                   'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N',
                                   'charge': 1})
    spectrum2 = Spectrum(mz=np.array([538.003174, 539.217773, 556.030396,
                                      599.352783, 851.380859, 852.370605,
                                      909.424438, 953.396606, 963.686768,
                                      964.524658
                                      ],
                                     dtype="float"),
                         intensities=np.array([0.28046377, 0.28900242,
                                               0.31933114, 0.32199162,
                                               0.34214536, 0.35616456,
                                               0.36216307, 0.41616014,
                                               0.71323034, 1.],
                                              dtype="float"),
                         metadata={'pepmass': (928.0, None),
                                   'spectrumid': 'CCMSLIB00000001761',
                                   'precursor_mz': 928.0,
                                   'parent_mass': 905.010782,
                                   'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N',
                                   'charge': 1})
    return [spectrum1, spectrum2]


def test_ms2library_set_settings(file_names):
    """Tests creating a ms2library object"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = file_names

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name,
                              cosine_score_tolerance=0.2)

    assert test_library.settings["cosine_score_tolerance"] == 0.2, \
        "Different value for attribute was expected"
    assert test_library.settings["base_nr_mass_similarity"] == 0.8, \
        "Different value for attribute was expected"


def test_select_best_matches():
    # todo add this testfunction, once the best filter step has been selected
    pass


def test_select_potential_true_matches(file_names, test_spectra):
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = file_names

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name)

    results = \
        test_library.select_potential_true_matches(test_spectra,
                                                   mass_tolerance=30,
                                                   s2v_score_threshold=0.6)
    assert isinstance(results, list), "Expected List"
    for df in results:
        assert isinstance(df, pd.DataFrame), "Expected a list with dataframes"
    expected_df_1 = pd.DataFrame(
        data={"spectrum_id": ['CCMSLIB00000001631', 'CCMSLIB00000001633',
                               'CCMSLIB00000001648', 'CCMSLIB00000001650'],
              "s2v_score": [0.910427, 0.973853, 0.978485, 0.979844],
              "parent_mass_difference":
                  [878.453724,878.453782,878.332724,878.410782]})
    expected_df_2 = pd.DataFrame(
        data={"spectrum_id": ["CCMSLIB00000001548"],
              "s2v_score": [0.995251],
              "parent_mass_difference": [939.242724]})
    pd.testing.assert_frame_equal(results[0],
                                  expected_df_1.set_index("spectrum_id"))
    pd.testing.assert_frame_equal(results[1],
                                  expected_df_2.set_index("spectrum_id"))


def test_analog_search(file_names, test_spectra):
    """Test analog search"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = file_names

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name)
    ms2q_model_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/ms2query_model_all_scores_dropout_regularization.hdf5')
    cutoff = 20
    result = test_library.analog_search(test_spectra, ms2q_model_file_name, cutoff)

    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_ms2library/expected_analog_search_results.pickle"))
    for i in range(len(expected_result)):
        assert expected_result[i].__eq__(result[i]), "different result table was expected"


def test_calculate_scores_for_metadata(file_names, test_spectra):
    """Test collect_matches_data_multiple_spectra method of ms2library"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = file_names

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name)

    ms2dscores:pd.DataFrame = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_ms2ds_scores.pickle'))
    results_table = ResultsTable(
        preselection_cut_off=20,
        ms2deepscores=ms2dscores.iloc[:, 0],
        query_spectrum=test_spectra[0],
        sqlite_file_name=sqlite_file_loc)
    results_table = test_library._calculate_scores_for_metascore(results_table)

    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_ms2library/expected_results_table_with_scores.pickle"))
    results_table.__eq__(expected_result)


def test_get_all_ms2ds_scores(file_names, test_spectra):
    """Test get_all_ms2ds_scores method of ms2library"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = file_names

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name)

    result = test_library._get_all_ms2ds_scores(test_spectra)

    expected_result:pd.DataFrame = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_ms2ds_scores.pickle'))
    assert isinstance(result, pd.DataFrame), "Expected dictionary"
    assert_frame_equal(result, expected_result)


def test_get_s2v_scores(file_names, test_spectra):
    """Test _get_s2v_scores method of MS2Library"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = file_names

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name)
    result = test_library._get_s2v_scores(
        test_spectra[0], ["CCMSLIB00000001572", "CCMSLIB00000001648"])
    assert np.allclose(result, np.array([0.97565603, 0.97848464])), \
        "Expected different Spec2Vec scores"


def test_get_average_ms2ds_for_inchikey14(file_names):
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = file_names

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name)
    inchickey14s = {"BKUKTJSDOUXYFL", "BTVYFIMKUHNOBZ"}
    ms2ds_scores = pd.Series(
        [0.1, 0.8, 0.3],
        index=['CCMSLIB00000001678',
               'CCMSLIB00000001651', 'CCMSLIB00000001653'])
    results = test_library._get_average_ms2ds_for_inchikey14(
        ms2ds_scores, inchickey14s)
    assert results == \
           {'BKUKTJSDOUXYFL': (0.1, 1), 'BTVYFIMKUHNOBZ': (0.55, 2)}, \
           "Expected different results"


def test_get_chemical_neighbourhood_scores(file_names):
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = file_names

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name)
    average_inchickey_scores = \
        {'BKUKTJSDOUXYFL': (0.8, 3),
         'UZMVEOVJASEKLP': (0.8, 2),
         'QWSYKJZSJYRUSS': (0.8, 2),
         'GRVRRAOIXXYICO': (0.8, 7),
         'WXDBUBIFYCCNLE': (0.8, 2),
         'ORRFIXSGNXBETO': (0.8, 2),
         'LLWMPGSQZXZZAE': (0.8, 4),
         'CTBBEXWJRAPJIZ': (0.8, 2),
         'YQLQWGVOWKPLFR': (0.8, 2),
         'BTVYFIMKUHNOBZ': (0.8, 2)}

    results = test_library._get_chemical_neighbourhood_scores(
        {"BKUKTJSDOUXYFL"}, average_inchickey_scores)
    assert isinstance(results, dict), "expected a dictionary"
    assert len(results) == 1, "Expected different number of results in " \
                              "dictionary"
    assert 'BKUKTJSDOUXYFL' in results
    scores = results['BKUKTJSDOUXYFL']
    assert isinstance(scores, tuple)
    assert len(scores) == 3, "Expected three scores for each InChiKey"
    assert math.isclose(scores[0], 0.8)
    assert scores[1] == 28
    assert math.isclose(scores[2], 0.46646038479969587)


def test_get_ms2query_model_prediction_single_spectrum():
    results_table = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_ms2library/expected_results_table_with_scores.pickle"))
    ms2q_model_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/ms2query_model_all_scores_dropout_regularization.hdf5')
    ms2query_nn_model = load_nn_model(ms2q_model_file_name)
    results = get_ms2query_model_prediction_single_spectrum(results_table, ms2query_nn_model)

    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_ms2library/expected_analog_search_results.pickle"))[0]
    expected_result.assert_results_table_equal(results)
