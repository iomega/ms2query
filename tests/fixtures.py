import os

import pytest

from ms2query.query_from_sqlite_database import SqliteLibrary
from ms2query.ms2library import MS2Library

@pytest.fixture(scope="package")
def path_to_general_test_files() -> str:
    return os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files')


@pytest.fixture(scope="package")
def path_to_test_files():
    return os.path.join(os.path.split(os.path.dirname(__file__))[0], 'tests/test_files')


@pytest.fixture(scope="package")
def sqlite_library(path_to_test_files):
    path_to_library = os.path.join(path_to_test_files, "general_test_files", "100_test_spectra.sqlite")
    return SqliteLibrary(path_to_library)

@pytest.fixture
def ms2library() -> MS2Library:
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
    ms2q_model_file_name = os.path.join(path_to_tests_dir,
        "general_test_files", "test_ms2q_rf_model.onnx")
    ms2library = MS2Library(sqlite_file_loc, spec2vec_model_file_loc, ms2ds_model_file_name,
                            s2v_pickled_embeddings_file, ms2ds_embeddings_file_name, ms2q_model_file_name,
                            spectrum_id_column_name=spectrum_id_column_name)
    return ms2library