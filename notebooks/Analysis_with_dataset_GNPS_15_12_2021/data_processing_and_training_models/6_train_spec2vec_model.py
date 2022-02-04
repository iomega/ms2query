import os
from matchms.filtering import (add_losses, add_precursor_mz, default_filters,
                               normalize_intensities,
                               reduce_to_number_of_peaks,
                               require_minimum_number_of_peaks, select_by_mz)
from ms2query.utils import load_pickled_file
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model


def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = default_filters(s)
    s = add_precursor_mz(s)
    s = normalize_intensities(s)
    s = reduce_to_number_of_peaks(s, n_required=5, ratio_desired=0.5, n_max=500)
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
    s = require_minimum_number_of_peaks(s, n_required=5)
    return s


# path_root = os.path.dirname(os.getcwd())
# path_data = os.path.join(os.path.dirname(path_root), "data/gnps_15_12_2021/")
path_data = "C:\\HSD\\OneDrive - Hochschule Düsseldorf\\Data\\ms2query"

training_spectra_annotated = load_pickled_file(os.path.join(path_data,
                                                            "GNPS_15_12_2021_pos_train.pickle"))
training_spectra_not_annotated = load_pickled_file(os.path.join(path_data,
                                                                "ALL_GNPS_15_12_2021_positive_not_annotated.pickle"))
all_spectra = training_spectra_annotated + training_spectra_not_annotated
# Load data from pickled file and apply filters
cleaned_spectra = [spectrum_processing(s) for s in all_spectra]

# Omit spectrums that didn't qualify for analysis
cleaned_spectra = [s for s in cleaned_spectra if s is not None]

# Create spectrum documents
reference_documents = [SpectrumDocument(s, n_decimals=2) for s in cleaned_spectra]

model_file = os.path.join(path_data, "trained_models",
                          "spec2vec_model_GNPS_15_12_2021.model")
model = train_new_word2vec_model(reference_documents,
                                 iterations=[10, 20, 30],
                                 filename=model_file,
                                 workers=4,
                                 progress_logger=True)
