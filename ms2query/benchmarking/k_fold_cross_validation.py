"""
This script is not needed for normally running MS2Query, instead it was used to split data for 20 fold
cross validation for the MS2Query manuscript
"""
import os
import sys
from typing import List
from matchms import Spectrum
from ms2query.create_new_library.split_data_for_training import split_spectra_in_random_inchikey_sets
from ms2query.utils import load_matchms_spectrum_objects_from_file, save_pickled_file
from ms2query.create_new_library.train_models import train_all_models
from ms2query.benchmarking.collect_test_data_results import generate_test_results
from ms2query.clean_and_filter_spectra import clean_normalize_and_split_annotated_spectra
from ms2query.ms2library import create_library_object_from_one_dir


def split_k_fold_cross_validation(spectra: List[Spectrum],
                                  k: int,
                                  ion_mode,
                                  output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    annotated_spectra, unnnotated_spectra = clean_normalize_and_split_annotated_spectra(spectra,
                                                                                        ion_mode,
                                                                                        True)
    save_pickled_file(unnnotated_spectra,
                      os.path.join(output_folder, "unannotated_training_spectra.pickle"))
    # split_spectra_in k sets
    spectrum_sets = split_spectra_in_random_inchikey_sets(annotated_spectra, k)
    for i in range(k):
        training_spectra = []
        test_spectra = []
        for j, spectrum_set in enumerate(spectrum_sets):
            if j != i:
                training_spectra += spectrum_set
            else:
                test_spectra = spectrum_set
        folder_name = f"test_split_{i}"
        os.mkdir(os.path.join(output_folder, folder_name))
        save_pickled_file(training_spectra,
                          os.path.join(output_folder, folder_name, "annotated_training_spectra.pickle"))
        save_pickled_file(test_spectra,
                          os.path.join(output_folder, folder_name, "test_spectra.pickle"))


def train_models_and_test_result_from_k_fold_folder(k_fold_split_folder :str,
                                                    i: str):
    folder_name = f"test_split_{i}"
    output_folder = os.path.join(k_fold_split_folder, folder_name)
    models_folder = os.path.join(output_folder, "models")
    test_results_folder = os.path.join(output_folder, "test_results")

    # Load in spectra
    unannotated_training_spectra = load_matchms_spectrum_objects_from_file(
        os.path.join(k_fold_split_folder, "unannotated_training_spectra.pickle"))
    annotated_training_spectra = load_matchms_spectrum_objects_from_file(
        os.path.join(k_fold_split_folder, folder_name, "annotated_training_spectra.pickle"))
    test_spectra = load_matchms_spectrum_objects_from_file(
        os.path.join(k_fold_split_folder, folder_name, "test_spectra.pickle"))

    # Train all models
    train_all_models(annotated_training_spectra,
                     unannotated_training_spectra,
                     models_folder)

    # Generate test results
    ms2library = create_library_object_from_one_dir(models_folder)
    generate_test_results(ms2library,
                          annotated_training_spectra,
                          test_spectra,
                          test_results_folder)


if __name__ == "__main__":
    k_fold_split_number = sys.argv[1]
    train_models_and_test_result_from_k_fold_folder("../../data/libraries_and_models/gnps_01_11_2022/20_fold_splits",
                                                    k_fold_split_number)
