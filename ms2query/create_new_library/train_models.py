"""
This script is not needed for normally running MS2Query, it is only needed to to train
new models
"""

import os
from spec2vec.model_building import train_new_word2vec_model
from ms2query.create_new_library.train_ms2deepscore import train_ms2deepscore_wrapper
from ms2query.create_new_library.train_ms2query_model import train_ms2query_model
from ms2query.create_new_library.library_files_creator import LibraryFilesCreator
from ms2query.utils import save_pickled_file, load_matchms_spectrum_objects_from_file
from ms2query.clean_and_filter_spectra import create_spectrum_documents, clean_normalize_and_split_annotated_spectra


class SettingsTrainingModels:
    def __init__(self,
                 settings):
        default_settings = {"ms2ds_fraction_validation_spectra": 30,
                            "ms2ds_epochs": 150,
                            "spec2vec_iterations": 30,
                            "ms2query_fraction_for_making_pairs": 40}
        if settings:
            for setting in settings:
                assert setting in default_settings, \
                    f"Available settings are {default_settings.keys()}"
                default_settings[setting] = settings[setting]
        self.ms2ds_fraction_validation_spectra: float = default_settings["ms2ds_fraction_validation_spectra"]
        self.ms2ds_epochs: int = default_settings["ms2ds_epochs"]
        self.ms2query_fraction_for_making_pairs: int = default_settings["ms2query_fraction_for_making_pairs"]
        self.spec2vec_iterations = default_settings["spec2vec_iterations"]


def train_all_models(annotated_training_spectra,
                     unannotated_training_spectra,
                     output_folder,
                     other_settings: dict = None):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    settings = SettingsTrainingModels(other_settings)
    # set file names of new generated files
    ms2deepscore_model_file_name = os.path.join(output_folder, "ms2deepscore_model.hdf5")
    spec2vec_model_file_name = os.path.join(output_folder, "spec2vec_model.model")
    ms2query_model_file_name = os.path.join(output_folder, "ms2query_model.pickle")
    ms2ds_history_figure_file_name = os.path.join(output_folder, "ms2deepscore_training_history.svg")

    # Train MS2Deepscore model
    train_ms2deepscore_wrapper(annotated_training_spectra,
                               ms2deepscore_model_file_name,
                               settings.ms2ds_fraction_validation_spectra,
                               settings.ms2ds_epochs,
                               ms2ds_history_figure_file_name
                               )

    # Train Spec2Vec model
    spectrum_documents = create_spectrum_documents(annotated_training_spectra + unannotated_training_spectra)

    train_new_word2vec_model(spectrum_documents,
                             iterations=settings.spec2vec_iterations,
                             filename=spec2vec_model_file_name,
                             workers=4,
                             progress_logger=True)

    # Train MS2Query model
    ms2query_model = train_ms2query_model(annotated_training_spectra,
                                          os.path.join(output_folder, "library_for_training_ms2query"),
                                          ms2deepscore_model_file_name,
                                          spec2vec_model_file_name,
                                          fraction_for_training=settings.ms2query_fraction_for_making_pairs)

    save_pickled_file(ms2query_model, ms2query_model_file_name)

    # Create library with all training spectra
    library_files_creator = LibraryFilesCreator(annotated_training_spectra,
                                                output_folder,
                                                spec2vec_model_file_name,
                                                ms2deepscore_model_file_name)
    library_files_creator.create_all_library_files()


def clean_and_train_models(spectrum_file: str,
                           ion_mode: str,
                           output_folder,
                           model_train_settings = None):
    """Trains a new MS2Deepscore, Spec2Vec and MS2Query model and creates all needed library files

    :param spectrum_file:
        The file name of the library spectra
    :param ion_mode:
        The ion mode of the spectra you want to use for training the models, choose from "positive" or "negative"
    :param output_folder:
        The folder in which the models and library files are stored.
    """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    assert os.path.isdir(output_folder), "The specified folder is not a folder"
    assert ion_mode in {"positive", "negative"}, "ion_mode should be set to 'positive' or 'negative'"

    spectra = load_matchms_spectrum_objects_from_file(spectrum_file)
    annotated_spectra, unnnotated_spectra = clean_normalize_and_split_annotated_spectra(spectra,
                                                                                        ion_mode,
                                                                                        do_pubchem_lookup=True)
    train_all_models(annotated_spectra,
                     unnnotated_spectra,
                     output_folder,
                     model_train_settings)


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "../../data/")
    spectra = load_matchms_spectrum_objects_from_file(
        os.path.join(data_folder,
                     "libraries_and_models/gnps_15_12_2021/in_between_files/ALL_GNPS_15_12_2021_raw_spectra.pickle"))
    spectra = spectra[:2000]

    train_all_models(spectra,
                     "positive",
                     "../../data/test_dir/test_train_all_models",
                     {"ms2ds_fraction_validation_spectra": 5,
                      "ms2ds_epochs": 10,
                      "spec2vec_iterations": 5,
                      "ms2query_fraction_for_making_pairs": 40}
                     )
