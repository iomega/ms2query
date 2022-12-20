"""
This script is not needed for normally running MS2Query, instead it was used to split data for 20 fold
cross validation for the MS2Query manuscript
"""
import os
import random
from typing import List
from matchms import Spectrum
from ms2query.create_new_library.split_data_for_training import split_spectra_in_random_inchikey_sets, select_spectra_per_unique_inchikey
from ms2query.utils import load_matchms_spectrum_objects_from_file, save_pickled_file
from ms2query.create_new_library.train_models import train_all_models
from ms2query.benchmarking.collect_test_data_results import generate_test_results, generate_exact_matches_test_results
from ms2query.clean_and_filter_spectra import clean_normalize_and_split_annotated_spectra
from ms2query.ms2library import create_library_object_from_one_dir


def split_and_store_annotated_unannotated(spectrum_file_name,
                                          ion_mode,
                                          output_folder):
    assert os.path.isdir(output_folder)
    assert not os.path.exists(os.path.join(output_folder, "unannotated_training_spectra.pickle"))
    assert not os.path.exists(os.path.join(output_folder, "annotated_training_spectra.pickle"))
    spectra = load_matchms_spectrum_objects_from_file(spectrum_file_name)
    annotated_spectra, unannotated_spectra = clean_normalize_and_split_annotated_spectra(spectra,
                                                                                        ion_mode,
                                                                                        True)
    save_pickled_file(unannotated_spectra,
                      os.path.join(output_folder, "unannotated_training_spectra.pickle"))
    save_pickled_file(annotated_spectra,
                      os.path.join(output_folder, "annotated_training_spectra.pickle"))
    return annotated_spectra, unannotated_spectra


def split_k_fold_cross_validation_analogue_test_set(annotated_spectra: List[Spectrum],
                                                    k: int,
                                                    output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
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


def split_k_fold_cross_validation_exact_match_test_set(annotated_spectra: List[Spectrum],
                                                       k: int,
                                                       output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    spectra_per_unique_inchikey = select_spectra_per_unique_inchikey(annotated_spectra)
    for i in range(k):
        training_spectra = []
        test_spectra = []
        for _, spectra in spectra_per_unique_inchikey.items():
            if len(spectra) == 1:
                training_spectra += spectra
            else:
                random.shuffle(spectra)
                test_spectra.append(spectra[0])
                training_spectra += spectra[1:]

        folder_name = f"test_split_{i}"
        os.mkdir(os.path.join(output_folder, folder_name))
        save_pickled_file(training_spectra,
                          os.path.join(output_folder, folder_name, "annotated_training_spectra.pickle"))
        save_pickled_file(test_spectra,
                          os.path.join(output_folder, folder_name, "test_spectra.pickle"))


def train_models_and_test_result_from_k_fold_folder(k_fold_split_folder:str,
                                                    i: str,
                                                    exact_matches=False):
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
    if not exact_matches:
        generate_exact_matches_test_results(ms2library,
                                            annotated_training_spectra,
                                            test_spectra,
                                            test_results_folder)
    else:
        generate_test_results(ms2library,
                              annotated_training_spectra,
                              test_spectra,
                              test_results_folder)
