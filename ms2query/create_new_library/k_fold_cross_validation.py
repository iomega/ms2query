import os
from typing import List
from matchms import Spectrum
import itertools
from ms2query.create_new_library.split_data_for_training import split_spectra_in_random_inchikey_sets
from ms2query.create_new_library.train_ms2deepscore import train_ms2deepscore_wrapper
from ms2query.create_new_library.library_files_creator import LibraryFilesCreator
from ms2query.utils import load_matchms_spectrum_objects_from_file


def train_k_fold_cross_validation(spectra: List[Spectrum], k: int, ion_mode, output_folder):
    # todo Select positive mode
    # todo Clean spectra
    # todo Split annotated and not annotated
    unannotated_spectra = []
    annotated_spectra = []
    # split_spectra_in k sets
    spectrum_sets = split_spectra_in_random_inchikey_sets(annotated_spectra, k)
    for i in range(k):
        # todo Create a folder containing all models and files for this test set.
        test_set = spectrum_sets[i]
        training_sets = [spectrum_set for j, spectrum_set in spectrum_sets if j!=i]
        # todo save test and trianing sets.
        training_set = (list(itertools.chain.from_iterable(training_sets)))
        train_all_models(training_set, unannotated_spectra, output_folder)





if __name__ == "__main__":
    pass
    # Train MS2Deepscore and Spec2Vec on 4/5th training spectra
    # Train MS2Query with MS2Deepscore and Spec2Vec
    # Use test set on MS2Query to test performance
    # Store the complete model and the 5 data splits.
