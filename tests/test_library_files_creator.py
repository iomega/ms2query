import os
import pandas as pd
import pytest
from ms2query.create_new_library.library_files_creator import LibraryFilesCreator
from ms2query.utils import (load_matchms_spectrum_objects_from_file,
                            load_pickled_file)
from ms2query.clean_and_filter_spectra import normalize_and_filter_peaks


def test_give_already_used_file_name(tmp_path, path_to_general_test_files, hundred_test_spectra):
    already_existing_file = os.path.join(tmp_path, "ms2query_library.sqlite")
    with open(already_existing_file, "w") as file:
        file.write("test")

    with pytest.raises(AssertionError):
        LibraryFilesCreator(hundred_test_spectra, tmp_path)


def test_store_ms2ds_embeddings(tmp_path, path_to_general_test_files,
                                hundred_test_spectra):
    """Tests store_ms2ds_embeddings"""
    base_file_name = os.path.join(tmp_path, '100_test_spectra')
    library_spectra = [normalize_and_filter_peaks(s) for s in hundred_test_spectra if s is not None]
    test_create_files = LibraryFilesCreator(library_spectra, base_file_name,
                                            ms2ds_model_file_name=os.path.join(path_to_general_test_files,
                                                                               'ms2ds_siamese_210301_5000_500_400.hdf5'))
    test_create_files.store_ms2ds_embeddings()

    new_embeddings_file_name = os.path.join(base_file_name, "ms2ds_embeddings.pickle")
    assert os.path.isfile(new_embeddings_file_name), \
        "Expected file to be created"
    # Test if correct embeddings are stored
    embeddings = load_pickled_file(new_embeddings_file_name)
    expected_embeddings = load_pickled_file(os.path.join(
        path_to_general_test_files,
        "100_test_spectra_ms2ds_embeddings.pickle"))
    pd.testing.assert_frame_equal(embeddings, expected_embeddings,
                                  check_exact=False,
                                  atol=1e-5)


def test_store_s2v_embeddings(tmp_path, path_to_general_test_files, hundred_test_spectra):
    """Tests store_ms2ds_embeddings"""
    base_file_name = os.path.join(tmp_path, '100_test_spectra')
    library_spectra = [normalize_and_filter_peaks(s) for s in hundred_test_spectra if s is not None]
    test_create_files = LibraryFilesCreator(library_spectra, base_file_name,
                                            s2v_model_file_name=os.path.join(path_to_general_test_files,
                                                                             "100_test_spectra_s2v_model.model"))
    test_create_files.store_s2v_embeddings()

    new_embeddings_file_name = os.path.join(base_file_name, "s2v_embeddings.pickle")
    assert os.path.isfile(new_embeddings_file_name), \
        "Expected file to be created"
    embeddings = load_pickled_file(new_embeddings_file_name)
    expected_embeddings = load_pickled_file(os.path.join(
        path_to_general_test_files,
        "100_test_spectra_s2v_embeddings.pickle"))
    pd.testing.assert_frame_equal(embeddings, expected_embeddings,
                                  check_exact=False,
                                  atol=1e-5)


def test_create_sqlite_file(tmp_path, path_to_general_test_files, hundred_test_spectra):
    test_create_files = LibraryFilesCreator(
        hundred_test_spectra[:20], output_directory=os.path.join(tmp_path, '100_test_spectra'))
    test_create_files.create_sqlite_file()

