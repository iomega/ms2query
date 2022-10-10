import os
import pandas as pd
from typing import List
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from matchms import Spectrum, calculate_scores
from matchms.similarity import FingerprintSimilarity
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from tensorflow.keras.callbacks import (  # pylint: disable=import-error
    EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error
from ms2query.create_new_library.create_sqlite_database import select_inchi_for_unique_inchikeys, add_fingerprint
from ms2query.utils import load_pickled_file


def train_ms2ds_model(training_spectra,
                      validation_spectra,
                      tanimoto_df,
                      output_model_file_name):
    assert not os.path.isfile(output_model_file_name), "The MS2Deepscore output model file name already exists"
    # Bin training spectra
    spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                     allowed_missing_percentage=100.0)
    binned_spectrums_training = spectrum_binner.fit_transform(training_spectra)
    # Bin validation spectra using the binner based on the training spectra.
    # Peaks that do not occur in the training spectra will not be binned in the validaiton spectra.
    binned_spectrums_val = spectrum_binner.transform(validation_spectra)

    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))

    training_generator = DataGeneratorAllInchikeys(
        binned_spectrums_training,
        selected_inchikeys=list({s.get("inchikey")[:14] for s in training_spectra}),
        reference_scores_df=tanimoto_df,
        dim=len(spectrum_binner.known_bins), # The number of bins created
        same_prob_bins=same_prob_bins,
        num_turns=2,
        augment_noise_max=10,
        augment_noise_intensity=0.01)

    selected_inchikeys = np.unique([s.get("inchikey")[:14] for s in validation_spectra])

    validation_generator = DataGeneratorAllInchikeys(
        binned_spectrums_val,
        selected_inchikeys=list({s.get("inchikey")[:14] for s in binned_spectrums_val}),
        reference_scores_df=tanimoto_df,
        dim=len(spectrum_binner.known_bins),  # The number of bins created
        same_prob_bins=same_prob_bins,
        num_turns=10, # Number of pairs for each InChiKey14 during each epoch.
        # To prevent data augmentation
        augment_removal_max=0, augment_removal_intensity=0, augment_intensity=0, augment_noise_max=0, use_fixed_set=True
    )

    model = SiameseModel(spectrum_binner, base_dims=(500, 500), embedding_dim=200, dropout_rate=0.2)

    model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

    # Save best model and include early stopping
    checkpointer = ModelCheckpoint(filepath=output_model_file_name, monitor='val_loss', mode="min", verbose=1, save_best_only=True)
    earlystopper_scoring_net = EarlyStopping(monitor='val_loss', mode="min", patience=10, verbose=1)
    # Fit model and save history
    history = model.model.fit(training_generator, validation_data=validation_generator, epochs=10, verbose=1,
                              callbacks=[checkpointer, earlystopper_scoring_net])
    model.load_weights(output_model_file_name)
    model.save(os.path.join("../data/test_dir/test_library_files_creator", "final_ms2ds_model.hdf5"))
    return history.history


def calculate_tanimoto_scores(list_of_spectra: List[Spectrum]):
    spectra_with_most_frequent_inchi_per_inchikey, unique_inchikeys = select_inchi_for_unique_inchikeys(list_of_spectra)
    # Add fingerprints
    fingerprint_spectra = []
    for spectrum in tqdm(spectra_with_most_frequent_inchi_per_inchikey,
                         desc="Calculating fingerprints for tanimoto scores"):
        spectrum_with_fingerprint = add_fingerprint(spectrum,
                                                    fingerprint_type="daylight",
                                                    nbits=2048)
        fingerprint_spectra.append(spectrum_with_fingerprint)

        assert spectrum_with_fingerprint.get("fingerprint") is not None, \
            f"Fingerprint for 1 spectrum could not be set smiles is {spectrum.get('smiles')}, inchi is {spectrum.get('inchi')}"

    # Specify type and calculate similarities
    similarity_measure = FingerprintSimilarity("jaccard")
    tanimoto_scores = calculate_scores(fingerprint_spectra, fingerprint_spectra,
                                       similarity_measure,
                                       is_symmetric=True).scores
    tanimoto_df = pd.DataFrame(tanimoto_scores, index=unique_inchikeys, columns=unique_inchikeys)
    return tanimoto_df

if __name__ == "__main__":
    from ms2query.utils import load_matchms_spectrum_objects_from_file
    data_folder = os.path.join(os.getcwd(), "../../data/")
    spectra = load_matchms_spectrum_objects_from_file(os.path.join(data_folder,
                                                                   "libraries_and_models/gnps_15_12_2021/in_between_files/ALL_GNPS_15_12_2021_positive_annotated.pickle"))
    train_spectra = spectra[:1000]
    validation_spectra = spectra[-1000:]
    print("loaded_spectra")

    tanimoto_df = load_pickled_file(os.path.join(data_folder,
                                                 "libraries_and_models/gnps_15_12_2021/in_between_files/GNPS_15_12_2021_pos_tanimoto_scores.pickle"))
    print("loaded_df")
    model_file_name = os.path.join("../data/test_dir/test_library_files_creator", "new_ms2ds_model.hdf5")
    train_ms2ds_model(train_spectra, validation_spectra, tanimoto_df, model_file_name)
