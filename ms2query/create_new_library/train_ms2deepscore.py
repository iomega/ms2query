import os
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
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
from ms2query.create_new_library.clean_and_split_data_for_training import split_spectra_on_inchikeys


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
    model.save(output_model_file_name)
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


def plot_history(history: Dict[str, List[float]]):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def train_ms2deepscore_wrapper(spectra: List[Spectrum], output_model_file_name, fraction_validation_inchikeys=30):
    assert not os.path.isfile(output_model_file_name), "The MS2Deepscore output model file name already exists"
    training_spectra, validation_spectra = split_spectra_on_inchikeys(spectra, fraction_validation_inchikeys)
    tanimoto_score_df = calculate_tanimoto_scores(spectra)
    history = train_ms2ds_model(training_spectra, validation_spectra, tanimoto_score_df, output_model_file_name)
    print(f"The training history is: {history}")
    plot_history(history)


if __name__ == "__main__":
    from ms2query.utils import load_matchms_spectrum_objects_from_file
    data_folder = os.path.join(os.getcwd(), "../../data/")
    spectra = load_matchms_spectrum_objects_from_file(os.path.join(data_folder,
                                                                   "libraries_and_models/gnps_15_12_2021/in_between_files/ALL_GNPS_15_12_2021_positive_annotated.pickle"))
    spectra = spectra[:1000]
    model_file_name = os.path.join("../../data/test_dir/test_library_files_creator", "new_ms2ds_model.hdf5")
    train_ms2deepscore_wrapper(spectra, model_file_name, 5)
