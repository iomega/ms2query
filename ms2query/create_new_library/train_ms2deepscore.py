"""
This script is not needed for normally running MS2Query, it is only needed to to train
new models
"""

import os
from typing import List
from matchms import Spectrum
from ms2deepscore import SpectrumBinner
from ms2deepscore.train_new_model.train_ms2deepscore import (plot_history,
                                                             train_ms2ds_model)
from ms2query.create_new_library.calculate_tanimoto_scores import \
    calculate_tanimoto_scores_unique_inchikey
from ms2query.create_new_library.split_data_for_training import \
    split_spectra_on_inchikeys


def train_ms2deepscore_wrapper(spectra: List[Spectrum],
                               output_model_file_name,
                               fraction_validation_spectra,
                               epochs,
                               ms2ds_history_file_name=None):
    assert not os.path.isfile(output_model_file_name), "The MS2Deepscore output model file name already exists"
    training_spectra, validation_spectra = split_spectra_on_inchikeys(spectra,
                                                                      fraction_validation_spectra)
    tanimoto_score_df = calculate_tanimoto_scores_unique_inchikey(spectra, spectra)
    spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                     allowed_missing_percentage=100.0)
    binned_spectrums_training = spectrum_binner.fit_transform(training_spectra)
    # Bin validation spectra using the binner based on the training spectra.
    # Peaks that do not occur in the training spectra will not be binned in the validaiton spectra.
    binned_spectrums_val = spectrum_binner.transform(validation_spectra)

    history = train_ms2ds_model(
            binned_spectrums_training,
            binned_spectrums_val,
            spectrum_binner,
            tanimoto_score_df,
            output_model_file_name,
            epochs=epochs,
            base_dims=(500, 500),
            embedding_dim=200,
    )

    print(f"The training history is: {history}")
    plot_history(history, ms2ds_history_file_name)
