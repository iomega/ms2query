import os
import sys
import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from onnxruntime import InferenceSession
from ms2query.create_new_library.train_ms2query_model import (
    DataCollectorForTraining, calculate_tanimoto_scores_with_library,
    convert_to_onnx_model, train_ms2query_model, train_random_forest)
from ms2query.ms2library import MS2Library
from ms2query.utils import predict_onnx_model


if sys.version_info < (3, 8):
    pass
else:
    pass


def test_data_collector_for_training_init(ms2library):
    """Tests if an object DataCollectorForTraining can be created"""
    DataCollectorForTraining(ms2library)


def test_get_matches_info_and_tanimoto(ms2library, hundred_test_spectra):
    preselection_cut_off = 2
    select_data_for_training = DataCollectorForTraining(ms2library, preselection_cut_off=preselection_cut_off)
    training_scores, training_labels = select_data_for_training.get_matches_info_and_tanimoto(hundred_test_spectra)
    assert isinstance(training_scores, pd.DataFrame)
    assert isinstance(training_labels, pd.DataFrame)
    assert training_scores.shape == (preselection_cut_off * len(hundred_test_spectra), 5)
    assert training_labels.shape == (preselection_cut_off * len(hundred_test_spectra), 1)
    assert list(training_scores.columns) == ['precursor_mz_library_spectrum',
                                             'precursor_mz_difference',
                                             's2v_score',
                                             'average_ms2deepscore_multiple_library_structures',
                                             'average_tanimoto_score_library_structures']
    assert list(training_labels.columns) == ['Tanimoto_score']
    assert round(training_scores.loc[0, "average_tanimoto_score_library_structures"], ndigits=5) == 0.57879


def test_calculate_all_tanimoto_scores(tmp_path, ms2library):
    query_spectrum = Spectrum(mz=np.array([], dtype="float"),
                              intensities=np.array([], dtype="float"),
                              metadata={"smiles": "CC1=CC(=O)O[C@@H](CCC[C@@H](CCCC(CCC[C@@H](CCC[C@H](CC(C[C@@H](CCCC(CCC[C@H](C[C@H](CCCCC1)O)O)O)O)O)O)O)O)O)C(C)(C)C"},
                              metadata_harmonization=False)
    spectra_ids_list = \
        [38, 3, 60]
    result = calculate_tanimoto_scores_with_library(ms2library.sqlite_library, query_spectrum, spectra_ids_list)
    expected_result = pd.DataFrame([0.199695, 0.177669, 0.192504],
                                   index=spectra_ids_list,
                                   columns=["Tanimoto_score"])
    assert isinstance(result, pd.DataFrame), "Expected a pd.Dataframe"
    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)


def test_train_and_save_random_forest(ms2library, hundred_test_spectra):
    select_data_for_training = DataCollectorForTraining(ms2library, preselection_cut_off=1)
    training_scores, training_labels = select_data_for_training.get_matches_info_and_tanimoto(hundred_test_spectra)

    ms2query_model = train_random_forest(training_scores, training_labels)
    onnx_model = convert_to_onnx_model(ms2query_model)
    onnx_model_session = InferenceSession(onnx_model.SerializeToString())
    predictions_onnx_model = predict_onnx_model(onnx_model_session, training_scores.values)

    # check if saving onnx model works
    predictions_sklearn_model = ms2query_model.predict(training_scores.values.astype(np.float32))
    assert np.allclose(predictions_onnx_model, predictions_sklearn_model)


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
