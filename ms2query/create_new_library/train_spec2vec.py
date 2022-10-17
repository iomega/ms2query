import os
from typing import List
from matchms.filtering import (add_losses, add_precursor_mz, default_filters,
                               normalize_intensities,
                               reduce_to_number_of_peaks,
                               require_minimum_number_of_peaks, select_by_mz)
from matchms import Spectrum
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model
from ms2query.utils import load_pickled_file


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


def train_spec2vec_model(spectra: List[Spectrum],
                         output_model_file_name):
    assert not os.path.isfile(output_model_file_name), "The Spec2Vec output model file name already exists"

    # Load data from pickled file and apply filters
    cleaned_spectra = [spectrum_processing(s) for s in spectra]

    # Omit spectrums that didn't qualify for analysis
    cleaned_spectra = [s for s in cleaned_spectra if s is not None]

    # Create spectrum documents
    reference_documents = [SpectrumDocument(s, n_decimals=2) for s in cleaned_spectra]

    model = train_new_word2vec_model(reference_documents,
                                     iterations=30,
                                     filename=output_model_file_name,
                                     workers=4,
                                     progress_logger=True)
