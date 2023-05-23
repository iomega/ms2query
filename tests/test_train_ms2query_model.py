import os
import pytest
import sys
import pandas as pd
from ms2query.create_new_library.train_ms2query_model import \
    DataCollectorForTraining, calculate_tanimoto_scores_with_library, train_random_forest, train_ms2query_model
from ms2query.utils import load_pickled_file, load_matchms_spectrum_objects_from_file, convert_to_onnx_model
from onnxruntime import InferenceSession
from ms2query.utils import predict_onnx_model
from ms2query.ms2library import MS2Library


if sys.version_info < (3, 8):
    pass
else:
    pass


@pytest.fixture
def path_to_test_dir():
    return os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files')


@pytest.fixture
def ms2library(path_to_test_dir):
    path_to_general_tests_dir = os.path.join(path_to_test_dir, 'general_test_files')

    return MS2Library(sqlite_file_name=os.path.join(path_to_general_tests_dir, "100_test_spectra.sqlite"),
                      s2v_model_file_name=os.path.join(path_to_general_tests_dir, "100_test_spectra_s2v_model.model"),
                      ms2ds_model_file_name=os.path.join(path_to_general_tests_dir,
                                                         "ms2ds_siamese_210301_5000_500_400.hdf5"),
                      pickled_s2v_embeddings_file_name=os.path.join(path_to_general_tests_dir,
                                                                    "100_test_spectra_s2v_embeddings.pickle"),
                      pickled_ms2ds_embeddings_file_name=os.path.join(path_to_general_tests_dir,
                                                                      "100_test_spectra_ms2ds_embeddings.pickle"),
                      ms2query_model_file_name=None)


@pytest.fixture
def query_spectra(path_to_test_dir):
    training_spectra_file_name = os.path.join(
        path_to_test_dir,
        "test_files_train_ms2query_nn/20_training_spectra.mgf")
    return load_matchms_spectrum_objects_from_file(training_spectra_file_name)


def test_data_collector_for_training_init(ms2library):
    """Tests if an object DataCollectorForTraining can be created"""
    DataCollectorForTraining(ms2library)


def test_get_matches_info_and_tanimoto(tmp_path, ms2library, query_spectra):
    select_data_for_training = DataCollectorForTraining(ms2library)
    result = select_data_for_training.get_matches_info_and_tanimoto(query_spectra)
    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_train_ms2query_nn",
        "expected_train_and_val_data.pickle"))[:2]
    assert isinstance(result, tuple), "Expected tuple to be returned"
    assert len(result) == 2, "Expected tuple to be returned"
    pd.testing.assert_frame_equal(result[0], expected_result[0], check_dtype=False, check_exact=False, rtol=1e-1)
    pd.testing.assert_frame_equal(result[1], expected_result[1], check_dtype=False, check_exact=False, rtol=1e-1)


def test_calculate_all_tanimoto_scores(tmp_path, ms2library, query_spectra):
    query_spectrum = query_spectra[0]
    spectra_ids_list = \
        [38, 3, 60]
    result = calculate_tanimoto_scores_with_library(ms2library.sqlite_library, query_spectrum, spectra_ids_list)
    expected_result = pd.DataFrame([0.199695, 0.177669, 0.192504],
                                   index=spectra_ids_list,
                                   columns=["Tanimoto_score"])
    assert isinstance(result, pd.DataFrame), "Expected a pd.Dataframe"
    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)


def test_train_random_forest():
    training_scores, training_labels = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_train_ms2query_nn",
        "expected_train_and_val_data.pickle"))[:2]
    ms2query_model = train_random_forest(training_scores, training_labels)
    onnx_model = convert_to_onnx_model(ms2query_model)
    onnx_model_session = InferenceSession(onnx_model.SerializeToString())
    predictions = predict_onnx_model(onnx_model_session, training_scores.values)


@pytest.mark.integration
def test_train_ms2query_model(path_to_general_test_files, tmp_path, hundred_test_spectra):
    models_folder = os.path.join(tmp_path, "models")
    ms2query_model = train_ms2query_model(
        training_spectra=hundred_test_spectra,
        library_files_folder=models_folder,
        ms2ds_model_file_name=os.path.join(path_to_general_test_files,
                                           "ms2ds_siamese_210301_5000_500_400.hdf5"),
        s2v_model_file_name=os.path.join(path_to_general_test_files,
                                         "100_test_spectra_s2v_model.model"),
        fraction_for_training=10
    )
