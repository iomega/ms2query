import os
import pandas as pd
import pickle
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


def train_ms2ds_model(training_spectra,
                      validation_spectra,
                      final_model_file_name,
                      checkpointer_model_file_name,
                      history_file_name):

    tanimoto_df = calculate_tanimoto_scores(training_spectra+validation_spectra)
    # Bin training spectra
    spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                     allowed_missing_percentage=100.0)
    binned_spectrums_training = spectrum_binner.fit_transform(training_spectra)
    # Bin validation spectra using the binner based on the training spectra.
    # Peaks that do not occur in the training spectra will not be binned in the validaiton spectra.
    binned_spectrums_val = spectrum_binner.transform(validation_spectra)

    dimension = len(spectrum_binner.known_bins)
    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))

    selected_inchikeys = np.unique([s.get("inchikey")[:14] for s in training_spectra])

    training_generator = DataGeneratorAllInchikeys(binned_spectrums_training, selected_inchikeys,
                                                   tanimoto_df,
                                                   dim=dimension,
                                                   same_prob_bins=same_prob_bins,
                                                   num_turns=2,
                                                   augment_noise_max=10,
                                                   augment_noise_intensity=0.01)

    selected_inchikeys = np.unique([s.get("inchikey")[:14] for s in validation_spectra])

    validation_generator = DataGeneratorAllInchikeys(binned_spectrums_val, selected_inchikeys,
                                                     tanimoto_df,
                                                     dim=dimension,
                                                     same_prob_bins=same_prob_bins,
                                                     num_turns=10,
                                                     augment_removal_max=0,
                                                     augment_removal_intensity=0,
                                                     augment_intensity=0,
                                                     augment_noise_max=0,
                                                     use_fixed_set=True)
    model = SiameseModel(spectrum_binner, base_dims=(500, 500), embedding_dim=200,
                         dropout_rate=0.2)
    # Save best model and include early stopping
    epochs = 150
    learning_rate = 0.001
    metrics = ["mae", tf.keras.metrics.RootMeanSquaredError()]

    # Parameters
    patience_scoring_net = 10

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=metrics)

    checkpointer = ModelCheckpoint(filepath=checkpointer_model_file_name, monitor='val_loss', mode="min", verbose=1, save_best_only=True)
    earlystopper_scoring_net = EarlyStopping(monitor='val_loss', mode="min", patience=patience_scoring_net, verbose=1)

    history = model.model.fit(training_generator, validation_data=validation_generator, epochs=epochs, verbose=1,
                              callbacks=[earlystopper_scoring_net, checkpointer,])

    # Save history
    with open(history_file_name, 'wb') as f:
        pickle.dump(history.history, f)

    model.save(final_model_file_name)
    # todo make sure the early stopping model is saved.


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
