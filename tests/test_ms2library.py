import os
import pickle
import pandas as pd
from gensim.models import Word2Vec
from ms2deepscore.models import load_model as load_ms2ds_model
from spec2vec import SpectrumDocument
from ms2query.ms2library import Ms2Library, create_spectrum_documents, \
    store_ms2ds_embeddings, store_s2v_embeddings
from ms2query.query_from_sqlite_database import get_spectra_from_sqlite


def test_create_spectrum_documents():
    path_to_pickled_file = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/first_10_spectra.pickle')
    with open(path_to_pickled_file, "rb") as pickled_file:
        spectrum_list = pickle.load(pickled_file)

    spectrum_documents, spectra_not_past_post_processing = \
        create_spectrum_documents(spectrum_list)
    assert isinstance(spectrum_documents, list), \
        "A list with spectrum_documents is expected"
    for spectrum_doc in spectrum_documents:
        assert isinstance(spectrum_doc, SpectrumDocument), \
            "A list with spectrum_documents is expected"


def test_store_ms2ds_embeddings(tmp_path):
    path_to_test_files_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files')
    pickled_file_loc = os.path.join(path_to_test_files_dir,
                                    "first_10_spectra.pickle")
    ms2ds_model_file_loc = os.path.join(
        path_to_test_files_dir,
        "ms2ds_siamese_210207_ALL_GNPS_positive_L1L2.hdf5")
    new_embeddings_file_name = os.path.join(tmp_path,
                                            "test_ms2ds_embeddings.pickle")
    control_embeddings_file_name = os.path.join(path_to_test_files_dir,
                                                "test_ms2ds_embeddings.pickle")
    # Load spectra and model in memory
    with open(pickled_file_loc, "rb") as pickled_file:
        spectrum_list = pickle.load(pickled_file)
    ms2ds_model = load_ms2ds_model(ms2ds_model_file_loc)

    # Create new pickled embeddings file
    store_ms2ds_embeddings(spectrum_list,
                           ms2ds_model,
                           new_embeddings_file_name)

    # Open new pickled embeddings file
    with open(new_embeddings_file_name, "rb") as new_embeddings_file:
        embeddings = pickle.load(new_embeddings_file)
    # Open control pickled embeddings file
    with open(control_embeddings_file_name, "rb") as control_embeddings_file:
        control_embeddings = pickle.load(control_embeddings_file)
    # Test if the correct embeddings are loaded into the new file.
    pd.testing.assert_frame_equal(embeddings, control_embeddings)


def test_store_s2v_embeddings(tmp_path):
    path_to_test_files_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files')
    pickled_file_loc = os.path.join(path_to_test_files_dir,
                                    "first_10_spectra.pickle")
    s2v_model_file_loc = os.path.join(
        path_to_test_files_dir,
        "spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")
    new_embeddings_file_name = os.path.join(tmp_path,
                                            "test_embeddings_file.pickle")
    control_embeddings_file_name = os.path.join(path_to_test_files_dir,
                                                "test_embeddings_file.pickle")
    # Load spectra and model in memory
    with open(pickled_file_loc, "rb") as pickled_file:
        spectrum_list = pickle.load(pickled_file)
    s2v_model = Word2Vec.load(s2v_model_file_loc)

    # Create new pickled embeddings file
    store_s2v_embeddings(spectrum_list,
                         s2v_model,
                         new_embeddings_file_name)
    # Open new pickled embeddings file
    with open(new_embeddings_file_name, "rb") as new_embeddings_file:
        embeddings = pickle.load(new_embeddings_file)
    # Open control pickled embeddings file
    with open(control_embeddings_file_name, "rb") as control_embeddings_file:
        control_embeddings = pickle.load(control_embeddings_file)
    # Test if the correct embeddings are loaded into the new file.
    pd.testing.assert_frame_equal(embeddings, control_embeddings)


def test_Ms2Library_set_settings():
    path_to_tests_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files')
    sqlite_file_loc = os.path.join(
        path_to_tests_dir,
        "test_spectra_database.sqlite")
    spec2vec_model_file_loc = os.path.join(
        path_to_tests_dir,
        "spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")
    s2v_pickled_embeddings_file = "../downloads/embeddings_all_spectra.pickle"
    ms2ds_model_file_name = "../../ms2deepscore/data/ms2ds_siamese_210207_ALL_GNPS_positive_L1L2.hdf5"
    ms2ds_embeddings_file_name = "../../ms2deepscore/data/ms2ds_embeddings_2_spectra.pickle"
    neural_network_model_file_location = "../model/nn_2000_queries_trimming_simple_10.hdf5"
    test_library = Ms2Library(sqlite_file_loc,
                              spec2vec_model_file_loc,
                              ms2ds_model_file_name,
                              s2v_pickled_embeddings_file,
                              ms2ds_embeddings_file_name,
                              neural_network_model_file_location,
                              cosine_score_tolerance=0.2)
    assert test_library.mass_tolerance == 1.0, \
        "Different value for attribute was expected"
    assert test_library.cosine_score_tolerance == 0.2, \
        "Different value for attribute was expected"
    assert test_library.base_nr_mass_similarity == 0.8, \
        "Different value for attribute was expected"

if __name__ == "__main__":
    test_create_Ms2Library()
