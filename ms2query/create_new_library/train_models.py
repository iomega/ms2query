"""
This script is not needed for normally running MS2Query, it is only needed to to train
new models
"""

import os
from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from spec2vec.model_building import train_new_word2vec_model
from ms2query.clean_and_filter_spectra import (
    clean_normalize_and_split_annotated_spectra, create_spectrum_documents)
from ms2query.create_new_library.library_files_creator import \
    LibraryFilesCreator
from ms2query.create_new_library.split_data_for_training import \
    split_spectra_on_inchikeys
from ms2query.create_new_library.train_ms2query_model import (
    convert_to_onnx_model, train_ms2query_model)
from ms2query.utils import load_matchms_spectrum_objects_from_file


class SettingsTrainingModels:
    def __init__(self,
                 settings: dict = None):
        default_settings = {"ms2ds_fraction_validation_spectra": 30,
                            "spec2vec_iterations": 30,
                            "ms2query_fraction_for_making_pairs": 40,
                            "add_compound_classes": True,
                            "ms2ds_training_settings": SettingsMS2Deepscore(
                                history_plot_file_name="ms2deepscore_training_history.svg",
                                model_file_name="ms2deepscore_model.pt",
                                epochs=150,
                                embedding_dim=400,
                                base_dims=(500, 500),
                                min_mz=10,
                                max_mz=1000,
                                mz_bin_width=0.1,
                                intensity_scaling=0.5
                            )}
        if settings:
            for setting in settings:
                assert setting in default_settings, \
                    f"Available settings are {default_settings.keys()}"
                default_settings[setting] = settings[setting]
        self.ms2ds_fraction_validation_spectra: float = default_settings["ms2ds_fraction_validation_spectra"]
        self.ms2ds_training_settings: SettingsMS2Deepscore = default_settings["ms2ds_training_settings"]
        self.ms2query_fraction_for_making_pairs: int = default_settings["ms2query_fraction_for_making_pairs"]
        self.spec2vec_iterations = default_settings["spec2vec_iterations"]
        self.add_compound_classes: bool = default_settings["add_compound_classes"]


def train_all_models(annotated_training_spectra,
                     unannotated_training_spectra,
                     output_folder,
                     settings: SettingsTrainingModels):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    # set file names of new generated files
    spec2vec_model_file_name = os.path.join(output_folder, "spec2vec_model.model")
    ms2query_model_file_name = os.path.join(output_folder, "ms2query_model.onnx")

    # Train MS2Deepscore model
    training_spectra, validation_spectra = split_spectra_on_inchikeys(annotated_training_spectra,
                                                                      settings.ms2ds_fraction_validation_spectra,
                                                                      )
    train_ms2ds_model(training_spectra, validation_spectra, output_folder,
                      settings.ms2ds_training_settings)

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
                                          os.path.join(output_folder, "ms2deepscore_model.pt"),
                                          spec2vec_model_file_name,
                                          fraction_for_training=settings.ms2query_fraction_for_making_pairs)
    convert_to_onnx_model(ms2query_model, ms2query_model_file_name)

    # Create library with all training spectra
    library_files_creator = LibraryFilesCreator(annotated_training_spectra,
                                                output_folder,
                                                spec2vec_model_file_name,
                                                os.path.join(output_folder, "ms2deepscore_model.pt"),
                                                add_compound_classes=settings.add_compound_classes)
    library_files_creator.create_all_library_files()


def clean_and_train_models(spectrum_file: str,
                           ion_mode: str,
                           output_folder,
                           model_train_settings=None,
                           do_pubchem_lookup = True):
    """Trains a new MS2Deepscore, Spec2Vec and MS2Query model and creates all needed library files

    :param spectrum_file:
        The file name of the library spectra
    :param ion_mode:
        The ion mode of the spectra you want to use for training the models, choose from "positive" or "negative"
    :param output_folder:
        The folder in which the models and library files are stored.
    :param model_train_settings:
        The settings used for training the models, options can be found in SettingsTrainingModels. If None is given
        all the default settings are used. The options and default settings are:
        {"ms2ds_fraction_validation_spectra": 30, "ms2ds_epochs": 150, "spec2vec_iterations": 30,
        "ms2query_fraction_for_making_pairs": 40, "add_compound_classes": True}
    :param do_pubchem_lookup:
        If True, the spectra are annotated with PubChem metadata before training the models.
    """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    assert os.path.isdir(output_folder), "The specified folder is not a folder"
    assert ion_mode in {"positive", "negative"}, "ion_mode should be set to 'positive' or 'negative'"

    settings = SettingsTrainingModels(model_train_settings)

    spectra = load_matchms_spectrum_objects_from_file(spectrum_file)
    annotated_spectra, unnnotated_spectra = clean_normalize_and_split_annotated_spectra(spectra,
                                                                                        ion_mode,
                                                                                        do_pubchem_lookup = do_pubchem_lookup)
    train_all_models(annotated_spectra,
                     unnnotated_spectra,
                     output_folder,
                     settings)
