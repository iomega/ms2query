import os
import pandas as pd
import pytest
import h5py
import keras
from ms2query.create_new_library.train_ms2deepscore import calculate_tanimoto_scores, train_ms2ds_model
from ms2query.utils import load_matchms_spectrum_objects_from_file, load_pickled_file


@pytest.fixture
def path_to_general_test_files() -> str:
    return os.path.join(
        os.getcwd(),
        './tests/test_files/general_test_files')


def test_calculate_tanimoto_scores(path_to_general_test_files):
    spectra = load_matchms_spectrum_objects_from_file(
        os.path.join(path_to_general_test_files, '100_test_spectra.pickle'))
    tanimoto_df = calculate_tanimoto_scores(spectra)
    expected_tanimoto_df = load_pickled_file(os.path.join(path_to_general_test_files,
                                                          "100_test_spectra_tanimoto_scores.pickle"))
    assert isinstance(tanimoto_df, pd.DataFrame), "Expected a pandas dataframe"
    pd.testing.assert_frame_equal(tanimoto_df, expected_tanimoto_df, check_exact=False, atol=1e-5)


def test_train_ms2ds_model(path_to_general_test_files, tmp_path):
    spectra = load_matchms_spectrum_objects_from_file(os.path.join(path_to_general_test_files, "100_test_spectra.pickle"))
    tanimoto_df = load_pickled_file(os.path.join(path_to_general_test_files, "100_test_spectra_tanimoto_scores.pickle"))
    model_file_name = os.path.join(tmp_path, "ms2ds_model.hdf5")
    history = train_ms2ds_model(spectra, spectra, tanimoto_df, model_file_name)
    assert os.path.isfile(model_file_name), "Expecte ms2ds model to be created and saved"
    with h5py.File(model_file_name, mode='r') as f:
        keras_model = keras.models.load_model(f)
    print(keras_model)
