import os
import pickle
import numpy as np
import tensorflow as tf
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from tensorflow.keras.callbacks import (  # pylint: disable=import-error
    EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

path_root = os.path.dirname(os.getcwd())

path_data = os.path.join(path_root, "../../../data/libraries_and_models/gnps_15_12_2021/in_between_files")


outfile = os.path.join(path_data, "GNPS_15_12_2021_neg_train.pickle")
with open(outfile, 'rb') as file:
    spectrums_training = pickle.load(file)
outfile = os.path.join(path_data, "GNPS_15_12_2021_neg_tanimoto_scores.pickle")
with open(outfile, 'rb') as file:
    tanimoto_df = pickle.load(file)
outfile = os.path.join(path_data, "GNPS_15_12_2021_neg_val_250_inchikeys.pickle")
with open(outfile, 'rb') as file:
    validation_spectra_250 = pickle.load(file)
outfile = os.path.join(path_data, "GNPS_15_12_2021_neg_val_3000_spectra.pickle")
with open(outfile, 'rb') as file:
    validation_spectra_3000 = pickle.load(file)

spectrums_val = validation_spectra_250 + validation_spectra_3000

spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                 allowed_missing_percentage=10.0)
binned_spectrums_training = spectrum_binner.fit_transform(spectrums_training)

binned_spectrums_val = spectrum_binner.transform(spectrums_val)

dimension = len(spectrum_binner.known_bins)
same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))

selected_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_training])

training_generator = DataGeneratorAllInchikeys(binned_spectrums_training, selected_inchikeys,
                                               tanimoto_df,
                                               dim=dimension,
                                               same_prob_bins=same_prob_bins,
                                               num_turns=2,
                                               augment_noise_max=10,
                                               augment_noise_intensity=0.01)
selected_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_val])

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
model = SiameseModel(spectrum_binner, base_dims=[500, 500], embedding_dim=200,
                     dropout_rate=0.2)
model.summary()
# Save best model and include early stopping
epochs = 150
learning_rate = 0.001
metrics = ["mae", tf.keras.metrics.RootMeanSquaredError()]

# Parameters
patience_scoring_net = 10
embedding_dim = 200
filename_base = "GNPS_15_12_2021"
model_output_file = os.path.join(path_data, "negative_mode_models", filename_base+".hdf5")

model.compile(
    loss='mse',
    optimizer=Adam(lr=learning_rate),
    metrics=metrics)

checkpointer = ModelCheckpoint(
    filepath = model_output_file,
    monitor='val_loss', mode="min",
    verbose=1,
    save_best_only=True
    )

earlystopper_scoring_net = EarlyStopping(
    monitor='val_loss', mode="min",
    patience=patience_scoring_net,
    verbose=1
    )

history = model.model.fit(training_generator,
    validation_data=validation_generator,
    epochs = epochs,
    verbose=1,
    callbacks = [
        earlystopper_scoring_net,
        checkpointer,
        ]
    )

# Save history
filename = os.path.join(path_data, filename_base+'_training_history.pickle')
with open(filename, 'wb') as f:
    pickle.dump(history.history, f)

model_output_file = os.path.join(path_data, "negative_mode_models", "ms2ds_"+filename_base+".hdf5")
model.save(model_output_file)
