import os
import pytest
from ms2query.create_new_library.train_models import train_all_models
from ms2query.utils import load_matchms_spectrum_objects_from_file


@pytest.fixture
def path_to_general_test_files() -> str:
    return os.path.join(
        os.getcwd(),
        '../tests/test_files/general_test_files')


def test_train_all_models(path_to_general_test_files, tmp_path):
    spectra = load_matchms_spectrum_objects_from_file(os.path.join(path_to_general_test_files, "100_test_spectra.pickle"))
    train_all_models(spectra*30, [], tmp_path, {"ms2ds_fraction_validation_spectra": 2,
                                                "ms2ds_epochs": 10})
