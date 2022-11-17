import os
import pytest
from ms2deepscore.models import SiameseModel
from ms2deepscore.models.load_model import load_model as load_ms2deepscore_model
from ms2query.create_new_library.train_ms2deepscore import train_ms2ds_model
from ms2query.utils import load_matchms_spectrum_objects_from_file, load_pickled_file
from tests.test_utils import path_to_general_test_files


def test_train_ms2ds_model(path_to_general_test_files, tmp_path):
    spectra = load_matchms_spectrum_objects_from_file(os.path.join(path_to_general_test_files, "100_test_spectra.pickle"))
    tanimoto_df = load_pickled_file(os.path.join(path_to_general_test_files, "100_test_spectra_tanimoto_scores.pickle"))
    model_file_name = os.path.join(tmp_path, "ms2ds_model.hdf5")
    epochs = 10
    history = train_ms2ds_model(spectra, spectra, tanimoto_df, model_file_name, epochs)
    assert os.path.isfile(model_file_name), "Expecte ms2ds model to be created and saved"
    ms2ds_model = load_ms2deepscore_model(model_file_name)
    assert isinstance(ms2ds_model, SiameseModel), "Expected a siamese model"
    assert isinstance(history, dict), "expected history to be a dictionary"
    assert list(history.keys()) == ['loss', 'mae', 'root_mean_squared_error', 'val_loss', 'val_mae', 'val_root_mean_squared_error']
    for scores in history.values():
        assert len(scores) == epochs, "expected the number of losses in the history to be equal to the number of epochs"
