import os
import pandas as pd
import pytest
from ms2query.library_files_creator import LibraryFilesCreator
from ms2query.utils import load_pickled_file
from .test_sqlite import check_sqlite_files_are_equal


@pytest.fixture
def path_to_general_test_files() -> str:
    return os.path.join(
        os.getcwd(),
        'tests/test_files/general_test_files')

def test_set_settings_correct(path_to_general_test_files):
    """Tests if settings are set correctly"""
    test_create_files = LibraryFilesCreator(os.path.join(
        path_to_general_test_files, '100_test_spectra.pickle'), output_base_filename="test_output_name",
        progress_bars=False)

    assert test_create_files.settings["output_file_sqlite"] == \
           "test_output_name.sqlite", "Expected different output_file_sqlite"
    assert test_create_files.settings["progress_bars"] is False, \
           "Expected different setting for progress_bar"
    assert test_create_files.settings["spectrum_id_column_name"] == \
           "spectrumid", "Expected different spectrum_id_column_name"
    assert test_create_files.settings["ms2ds_embeddings_file_name"] == \
           "test_output_name_ms2ds_embeddings.pickle", \
           "Expected different ms2ds_embeddings_file_name"
    assert test_create_files.settings["s2v_embeddings_file_name"] == \
           "test_output_name_s2v_embeddings.pickle", \
           "Expected different s2v_embeddings_file_name"


def test_set_settings_wrong():
    """Tests if an error is raised if a wrong attribute is passed"""
    pickled_spectra_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/100_test_spectra.pickle')
    pytest.raises(AssertionError, LibraryFilesCreator,
                  pickled_spectra_file_name, "output_filename",
                  not_recognized_attribute="test_value")


def test_give_already_used_file_name(tmp_path):
    base_file_name = os.path.join(tmp_path, "base_file_name")
    already_existing_file = base_file_name + ".sqlite"
    with open(already_existing_file, "w") as file:
        file.write("test")

    pickled_spectra_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/100_test_spectra.pickle')
    pytest.raises(AssertionError, LibraryFilesCreator,
                  pickled_spectra_file_name, base_file_name)


def test_store_ms2ds_embeddings(tmp_path, path_to_general_test_files):
    """Tests store_ms2ds_embeddings"""
    base_file_name = os.path.join(tmp_path, '100_test_spectra')
    test_create_files = LibraryFilesCreator(os.path.join(
        path_to_general_test_files, '100_test_spectra.pickle'), base_file_name,
        ms2ds_model_file_name=os.path.join(path_to_general_test_files, 'ms2ds_siamese_210301_5000_500_400.hdf5'))
    test_create_files.clean_spectra()
    test_create_files.store_ms2ds_embeddings()

    new_embeddings_file_name = base_file_name + "_ms2ds_embeddings.pickle"
    assert os.path.isfile(new_embeddings_file_name), \
        "Expected file to be created"
    # Test if correct embeddings are stored
    embeddings = load_pickled_file(new_embeddings_file_name)
    expected_embeddings = load_pickled_file(os.path.join(
        path_to_general_test_files,
        "test_files_without_spectrum_id",
        "100_test_spectra_ms2ds_embeddings.pickle"))
    pd.testing.assert_frame_equal(embeddings, expected_embeddings,
                                  check_exact=False,
                                  atol=1e-5)


def test_store_s2v_embeddings(tmp_path, path_to_general_test_files):
    """Tests store_ms2ds_embeddings"""
    base_file_name = os.path.join(tmp_path, '100_test_spectra')
    test_create_files = LibraryFilesCreator(os.path.join(
        path_to_general_test_files, '100_test_spectra.pickle'), base_file_name,
        s2v_model_file_name=os.path.join(path_to_general_test_files, "100_test_spectra_s2v_model.model"))
    test_create_files.clean_spectra()
    test_create_files.store_s2v_embeddings()

    new_embeddings_file_name = base_file_name + "_s2v_embeddings.pickle"
    assert os.path.isfile(new_embeddings_file_name), \
        "Expected file to be created"
    embeddings = load_pickled_file(new_embeddings_file_name)
    expected_embeddings = load_pickled_file(os.path.join(
        path_to_general_test_files,
        "test_files_without_spectrum_id",
        "100_test_spectra_s2v_embeddings.pickle"))
    pd.testing.assert_frame_equal(embeddings, expected_embeddings,
                                  check_exact=False,
                                  atol=1e-5)


def test_calculate_tanimoto_scores(tmp_path, path_to_general_test_files):
    base_file_name = os.path.join(tmp_path, '100_test_spectra')
    test_create_files = LibraryFilesCreator(os.path.join(
        path_to_general_test_files, '100_test_spectra.pickle'),
        base_file_name)
    test_create_files.calculate_tanimoto_scores()
    result: pd.DataFrame = test_create_files.tanimoto_scores
    result.sort_index(inplace=True)
    result.sort_index(1, inplace=True)
    expected_result = load_pickled_file(path_to_general_test_files + "/100_test_spectra_tanimoto_scores.pickle")
    pd.testing.assert_frame_equal(result, expected_result, check_exact=False, atol=1e-5)
