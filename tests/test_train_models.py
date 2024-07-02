import os
import string
import numpy as np
import pytest
from matchms.Spectrum import Spectrum
from ms2deepscore import SettingsMS2Deepscore
from ms2query.create_new_library.train_models import (SettingsTrainingModels,
                                                      train_all_models)
from ms2query.ms2library import MS2Library, create_library_object_from_one_dir


def create_test_spectra(num_of_unique_inchikeys):
    # Define other parameters
    mz, intens = 100.0, 0.1
    spectrums = []
    letters = list(string.ascii_uppercase[:num_of_unique_inchikeys])
    letters += letters

    # Create fake spectra
    fake_inchikeys = []
    for i, letter in enumerate(letters):
        dummy_inchikey = f"{14 * letter}-{10 * letter}-N"
        # fingerprint = generate_binary_vector(i)
        fake_inchikeys.append(dummy_inchikey)
        spectrums.append(
            Spectrum(mz=np.array([mz + (i + 1) * 1.0, mz + 100 + (i + 1) * 1.0, mz + 200 + (i + 1) * 1.0]),
                     intensities=np.array([intens, intens, intens]),
                     metadata={"precursor_mz": 111.1,
                               "inchikey": dummy_inchikey,
                               "smiles": "C"*(i+1)
                               }))
    return spectrums


@pytest.mark.integration
def test_train_all_models(tmp_path):
    test_spectra = create_test_spectra(11)

    models_folder = os.path.join(tmp_path, "models")
    train_all_models(test_spectra, test_spectra, output_folder=models_folder,
                     settings=SettingsTrainingModels({"ms2ds_fraction_validation_spectra": 2,
                                                      "ms2ds_training_settings": SettingsMS2Deepscore(
                                                          mz_bin_width=1.0,
                                                          epochs=2,
                                                          base_dims=(100, 100),
                                                          embedding_dim=50,
                                                          same_prob_bins=np.array([(0, 1.0)]),
                                                          average_pairs_per_bin=2,
                                                          batch_size=2),
                                                      "spec2vec_iterations": 2,
                                                      "ms2query_fraction_for_making_pairs": 10,
                                                      "add_compound_classes": False}))
    ms2library = create_library_object_from_one_dir(models_folder)
    assert isinstance(ms2library, MS2Library)
