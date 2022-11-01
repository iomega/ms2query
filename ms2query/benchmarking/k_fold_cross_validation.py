import os
from typing import List
from matchms import Spectrum
from ms2query.create_new_library.split_data_for_training import split_spectra_in_random_inchikey_sets
from ms2query.utils import load_matchms_spectrum_objects_from_file, save_pickled_file
from ms2query.create_new_library.train_models import train_all_models
from ms2query.benchmarking.collect_test_data_results import generate_test_results
from ms2query.clean_and_filter_spectra import clean_normalize_and_split_annotated_spectra


def split_k_fold_cross_validation(spectra: List[Spectrum],
                                  k: int,
                                  ion_mode,
                                  output_folder):
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


def train_models_and_test_result_from_k_fold_folder(k_fold_split_folder,
                                                    i):
    unannotated_training_spectra = load_matchms_spectrum_objects_from_file(
        os.path.join(k_fold_split_folder, "unannotated_training_spectra.pickle"))
    folder_name = f"test_split_{i}"
    annotated_training_spectra = load_matchms_spectrum_objects_from_file(
        os.path.join(k_fold_split_folder, folder_name, "annotated_training_spectra.pickle"))
    test_spectra = load_matchms_spectrum_objects_from_file(
        os.path.join(k_fold_split_folder, folder_name, "test_spectra.pickle"))
    train_models_and_create_test_results(annotated_training_spectra,
                                         unannotated_training_spectra,
                                         test_spectra,
                                         output_folder=os.path.join(k_fold_split_folder, folder_name))


def train_models_and_create_test_results(annotated_training_spectra: List[Spectrum],
                                         unannotated_training_spectra: List[Spectrum],
                                         test_spectra,
                                         output_folder: str
                                         ):
    models_folder = os.path.join(output_folder, "models")
    train_all_models(annotated_training_spectra,
                     unannotated_training_spectra,
                     models_folder
                     )

    test_results_file_name = os.path.join(output_folder,
                                          "test_results.json")
    generate_test_results(models_folder,
                          annotated_training_spectra,
                          test_spectra,
                          test_results_file_name)


if __name__ == "__main__":
    from ms2query.utils import load_matchms_spectrum_objects_from_file

    data_folder = os.path.join(os.getcwd(), "../../data/")
    # spectra = load_matchms_spectrum_objects_from_file(os.path.join(data_folder,
    #                                                                "libraries_and_models/gnps_15_12_2021/in_between_files/ALL_GNPS_15_12_2021_raw_spectra.pickle"))
    # spectra = spectra[:2000]
    # annotated_library_spectra, unnnotated_library_spectra = clean_normalize_and_split_annotated_spectra(spectra,
    #                                                                                                     "positive",
    #                                                                                                     True)
    # spectra = load_matchms_spectrum_objects_from_file("../../data/test_dir/test_train_all_models/training_spectra_used/cleaned_training_spectra.pickle")
    # test_spectra = spectra[:10]
    # annotated_library_spectra = spectra[10:-10]
    # unannotated_library_spectra = spectra[-10:]
    #
    # generate_test_results(os.path.join(data_folder, "test_dir/test_k_fold_cross_validation", "models"),
    #                       annotated_library_spectra,
    #                       test_spectra,
    #                       os.path.join(data_folder, "test_dir/test_k_fold_cross_validation", "test_results.json"))
    # train_models_and_create_test_results(annotated_library_spectra,
    #                                      unannotated_library_spectra,
    #                                      test_spectra,
    #                                      os.path.join(data_folder, "test_dir/test_k_fold_cross_validation"))
