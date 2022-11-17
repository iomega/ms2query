import os
import pytest
from ms2query.create_new_library.train_models import clean_and_train_models
from ms2query.ms2library import create_library_object_from_one_dir, MS2Library
from tests.test_utils import path_to_general_test_files


@pytest.mark.integration
def test_train_all_models(path_to_general_test_files, tmp_path):
    path_to_test_spectra = os.path.join(path_to_general_test_files, "1000_test_spectra.pickle")
    models_folder = os.path.join(tmp_path, "models")
    clean_and_train_models(path_to_test_spectra,
                           "positive",
                           models_folder,
                           {"ms2ds_fraction_validation_spectra": 2,
                            "ms2ds_epochs": 2,
                            "spec2vec_iterations": 2,
                            "ms2query_fraction_for_making_pairs": 400}
                           )
    ms2library = create_library_object_from_one_dir(models_folder)
    assert isinstance(ms2library, MS2Library)