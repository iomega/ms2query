import os
import numpy as np
from pandas.testing import assert_frame_equal
from matchms import Spectrum
from ms2query.ms2library import MS2Library, get_ms2query_model_prediction
from ms2query.app_helpers import load_pickled_file
from pandas import DataFrame


def test_ms2library_set_settings():
    """Tests creating a ms2library object"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = get_test_file_names()

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


def test_collect_matches_data_multiple_spectra():
    """Test collect_matches_data_multiple_spectra method of ms2library"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = get_test_file_names()

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name)

    test_spectra = create_test_spectra()

    result = test_library.get_analog_search_scores(test_spectra, 20)
    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_ms2library/expected_matches_with_averages.pickle"))
    assert isinstance(result, dict), "Expected dictionary"
    for key in result:
        assert isinstance(key, str), "Expected keys of dict to be string"
        assert_frame_equal(result[key], expected_result[key])
    # todo create new test file, once final decision is made about all
    #  scores calculated


def test_pre_select_spectra():
    """Test pre_select_spectra method of ms2library"""
    pass
    # todo change test, so it works with new splitted workflow
    # sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
    #     ms2ds_model_file_name, ms2ds_embeddings_file_name, \
    #     spectrum_id_column_name = get_test_file_names()
    #
    # test_library = MS2Library(sqlite_file_loc,
    #                           spec2vec_model_file_loc,
    #                           ms2ds_model_file_name,
    #                           s2v_pickled_embeddings_file,
    #                           ms2ds_embeddings_file_name,
    #                           spectrum_id_column_name=spectrum_id_column_name)
    #
    # test_spectra = create_test_spectra()
    #
    # preselected_spectra = test_library.pre_select_spectra(test_spectra)
    # expected_result = load_pickled_file(os.path.join(
    #     os.path.split(os.path.dirname(__file__))[0],
    #     'tests/test_files/test_files_ms2library',
    #     'expected_preselected_spectra.pickle'))
    # assert isinstance(preselected_spectra, dict), "Expected a dictionary"
    # assert isinstance(list(preselected_spectra.values())[0], list), \
    #     "Expected a dict with list as values"
    # assert isinstance(list(preselected_spectra.values())[0][0], str), \
    #     "Expected a dictionary with list with string"
    # assert isinstance(list(preselected_spectra.keys())[0], str), \
    #     "Expected dict with as keys str"
    # assert preselected_spectra == expected_result, \
    #     "Expected different preselected spectra"


def test_get_all_ms2ds_scores():
    """Test get_all_ms2ds_scores method of ms2library"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name = get_test_file_names()

    test_library = MS2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              spectrum_id_column_name=spectrum_id_column_name)
    test_spectra = create_test_spectra()
    result = test_library._get_all_ms2ds_scores(test_spectra)

    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_ms2ds_scores.pickle'))
    assert isinstance(result, DataFrame), "Expected dictionary"
    assert_frame_equal(result, expected_result)


def test_collect_data_for_ms2query_model():
    pass
    # todo rewrite test for new splitted workflow
    # """Test collect_data_for_ms2query_model method of ms2library"""
    # sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
    #     ms2ds_model_file_name, ms2ds_embeddings_file_name, \
    #     spectrum_id_column_name = get_test_file_names()
    #
    # test_library = MS2Library(sqlite_file_loc,
    #                           spec2vec_model_file_loc,
    #                           ms2ds_model_file_name,
    #                           s2v_pickled_embeddings_file,
    #                           ms2ds_embeddings_file_name,
    #                           spectrum_id_column_name=spectrum_id_column_name)
    #
    # test_spectrum = create_test_spectra()[0]
    # preselected_spectra = load_pickled_file(os.path.join(
    #     os.path.split(os.path.dirname(__file__))[0],
    #     'tests/test_files/test_files_ms2library',
    #     'expected_preselected_spectra.pickle'))
    # result = test_library._collect_data_for_ms2query_model(
    #     test_spectrum,
    #     preselected_spectra[test_spectrum.get(spectrum_id_column_name)])
    # expected_result = load_pickled_file(os.path.join(
    #     os.path.split(os.path.dirname(__file__))[0],
    #     'tests/test_files/test_files_ms2library/expected_matches_data.pickle')
    #     )[test_spectrum.get(spectrum_id_column_name)]
    # assert isinstance(result, DataFrame), "Expected dictionary"
    # assert_frame_equal(result, expected_result[["parent_mass",
    #                                            "mass_sim",
    #                                            "s2v_scores",
    #                                            "ms2ds_scores"]])


def test_get_ms2query_model_prediction():
    """Test get_ms2query_model_prediction method of ms2library"""
    matches_info = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_matches_data.pickle'))
    ms2q_model_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files/ms2query_model.hdf5')
    result = get_ms2query_model_prediction(matches_info,
                                           ms2q_model_file_name)
    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library',
        'expected_ms2query_model_scores.pickle'))
    assert isinstance(result, dict), "Expected dictionary"
    for key in result:
        assert isinstance(key, str), "Expected keys of dict to be string"
        assert_frame_equal(result[key], expected_result[key])


def get_test_file_names():
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


def create_test_spectra():
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
                                   'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N'})
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
                                   'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N'})
    return [spectrum1, spectrum2]

import pandas as pd
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 1000)
