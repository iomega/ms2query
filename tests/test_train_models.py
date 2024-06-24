import os
import numpy as np
import pytest
from ms2deepscore import SettingsMS2Deepscore

from ms2query.create_new_library.train_models import clean_and_train_models
from ms2query.ms2library import MS2Library, create_library_object_from_one_dir


@pytest.mark.integration
def test_train_all_models(path_to_general_test_files, tmp_path):
    path_to_test_spectra = os.path.join(path_to_general_test_files, "2000_negative_test_spectra.mgf")
    models_folder = os.path.join(tmp_path, "models")
    clean_and_train_models(path_to_test_spectra,
                           "negative",
                           models_folder,
                           {"ms2ds_fraction_validation_spectra": 2,
                            "ms2ds_training_settings": SettingsMS2Deepscore(
                                mz_bin_width=1.0,
                                epochs=2,
                                base_dims=(100, 100),
                                embedding_dim=50,
                                same_prob_bins=np.array([(0, 0.5), (0.5, 1.0)]),
                                average_pairs_per_bin=2,
                                batch_size=2),
                            "spec2vec_iterations": 2,
                            "ms2query_fraction_for_making_pairs": 400,
                            "add_compound_classes": False}
                           )
    ms2library = create_library_object_from_one_dir(models_folder)
    assert isinstance(ms2library, MS2Library)
