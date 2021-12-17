import os
from matchms.filtering import add_losses
from matchms.filtering import add_precursor_mz
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model
import pickle


def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = default_filters(s)
    s = add_precursor_mz(s)
    s = normalize_intensities(s)
    s = reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5, n_max=500)
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
    s = require_minimum_number_of_peaks(s, n_required=10)
    return s

path_root = os.path.dirname(os.getcwd())
path_data = os.path.join(os.path.dirname(path_root), "data/gnps_24_11_2021/")
training_spectra_annotated = pickle.load(open(os.path.join(path_data, "positive_mode/GNPS_24_11_2021_pos_train.pickle"), "rb"))
training_spectra_not_annotated = pickle.load(open(os.path.join(path_data, "in_between_files_data_split/ALL_GNPS_24_11_2021_positive_not_annotated.pickle"), "rb"))
all_spectra = training_spectra_annotated + training_spectra_not_annotated
# Load data from pickled file and apply filters
cleaned_spectra = [spectrum_processing(s) for s in all_spectra]

# Omit spectrums that didn't qualify for analysis
cleaned_spectra = [s for s in cleaned_spectra if s is not None]

# Create spectrum documents
reference_documents = [SpectrumDocument(s, n_decimals=2) for s in cleaned_spectra]

model_file = "spec2vec_model_GNPS_24_11_2021.model"
model = train_new_word2vec_model(reference_documents, iterations=[10, 20, 30], filename=model_file,
                                 workers=2, progress_logger=True)