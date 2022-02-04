import os
import sys
import pandas as pd
from ms2query.select_data_for_training_ms2query_nn import \
    DataCollectorForTraining
from ms2query.utils import load_pickled_file


if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


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
    training_spectra_file_name = os.path.join(
        path_to_tests_dir,
        "test_files_train_ms2query_nn/20_training_spectra.pickle")
    validation_spectra_file_name = os.path.join(
        path_to_tests_dir,
        "test_files_train_ms2query_nn/20_validation_spectra.pickle")
    tanimoto_scores_file_name = os.path.join(
        path_to_tests_dir,
        "general_test_files/100_test_spectra_tanimoto_scores.pickle"
        )
    return sqlite_file_loc, spec2vec_model_file_loc, \
        s2v_pickled_embeddings_file, ms2ds_model_file_name, \
        ms2ds_embeddings_file_name, spectrum_id_column_name, \
        training_spectra_file_name, validation_spectra_file_name, \
        tanimoto_scores_file_name


def test_data_collector_for_training_init():
    """Tests if an object DataCollectorForTraining can be created"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name, training_spectra_file_name, \
        testing_spectra_file_name, tanimoto_scores_file_name \
        = get_test_file_names()

    DataCollectorForTraining(sqlite_file_loc, spec2vec_model_file_loc,
                             ms2ds_model_file_name, s2v_pickled_embeddings_file,
                             ms2ds_embeddings_file_name,
                             training_spectra_file_name,
                             testing_spectra_file_name,
                             tanimoto_scores_file_name,
                             spectrum_id_column_name=spectrum_id_column_name)


def test_create_train_and_val_data_with_saving(tmp_path):
    """Test create_train_and_val_data without saving the files"""
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name, training_spectra_file_name, \
        validation_spectra_file_name, tanimoto_scores_file_name = \
        get_test_file_names()
    save_file_name = os.path.join(
        tmp_path, "test_training_and_validation_set_and_labels")

    select_data_for_training = DataCollectorForTraining(
        sqlite_file_loc, spec2vec_model_file_loc, ms2ds_model_file_name,
        s2v_pickled_embeddings_file, ms2ds_embeddings_file_name,
        training_spectra_file_name, validation_spectra_file_name,
        tanimoto_scores_file_name,
        spectrum_id_column_name=spectrum_id_column_name)
    returned_results = \
        select_data_for_training.create_train_and_val_data(
            save_file_name=save_file_name)
    assert os.path.exists(save_file_name), "Expected file to be created"

    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_train_ms2query_nn",
        "expected_train_and_val_data.pickle"))
    result_in_file = load_pickled_file(save_file_name)
    # Test if the right result is returned
    assert isinstance(returned_results, tuple), \
        "Expected a tuple to be returned"
    assert len(returned_results) == 4, "Expected a tuple with length 4"
    for i, result in enumerate(returned_results):
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected_result[i])
    # Test if right information is stored in file
    assert isinstance(result_in_file, tuple), \
        "Expected a tuple to be returned"
    assert len(result_in_file) == 4, "Expected a tuple with length 4"
    for i, result in enumerate(returned_results):
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected_result[i])


def test_get_matches_info_and_tanimoto():
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name, training_spectra_file_name, \
        validation_spectra_file_name, tanimoto_scores_file_name\
        = get_test_file_names()

    select_data_for_training = DataCollectorForTraining(
        sqlite_file_loc, spec2vec_model_file_loc, ms2ds_model_file_name,
        s2v_pickled_embeddings_file, ms2ds_embeddings_file_name,
        training_spectra_file_name, validation_spectra_file_name,
        tanimoto_scores_file_name,
        spectrum_id_column_name=spectrum_id_column_name)

    query_spectra = load_pickled_file(training_spectra_file_name)

    result = select_data_for_training.get_matches_info_and_tanimoto(
        query_spectra)
    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_train_ms2query_nn",
        "expected_train_and_val_data.pickle"))[:2]
    assert isinstance(result, tuple), "Expected tuple to be returned"
    assert len(result) == 2, "Expected tuple to be returned"
    pd.testing.assert_frame_equal(result[0], expected_result[0])
    pd.testing.assert_frame_equal(result[1], expected_result[1])


def test_get_tanimoto_for_spectrum_ids():
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name, training_spectra_file_name, \
        validation_spectra_file_name, tanimoto_scores_file_name \
        = get_test_file_names()

    select_data_for_training = DataCollectorForTraining(
        sqlite_file_loc, spec2vec_model_file_loc, ms2ds_model_file_name,
        s2v_pickled_embeddings_file, ms2ds_embeddings_file_name,
        training_spectra_file_name, validation_spectra_file_name,
        tanimoto_scores_file_name,
        spectrum_id_column_name=spectrum_id_column_name)

    query_spectrum = load_pickled_file(training_spectra_file_name)[0]
    spectra_ids_list = \
        ['CCMSLIB00000001603', 'CCMSLIB00000001652', 'CCMSLIB00000001640']
    result = select_data_for_training.get_tanimoto_for_spectrum_ids(
        query_spectrum,
        spectra_ids_list)
    expected_result = pd.DataFrame([0.199695, 0.177669, 0.192504],
                                   index=spectra_ids_list,
                                   columns=["Tanimoto_score"])
    assert isinstance(result, pd.DataFrame), "Expected a pd.Dataframe"
    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)
