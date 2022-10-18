import os
from ms2query.create_new_library.train_ms2deepscore import train_ms2deepscore_wrapper
from ms2query.create_new_library.train_spec2vec import train_spec2vec_model
from ms2query.create_new_library.train_ms2query_model import train_ms2query_model
from ms2query.create_new_library.library_files_creator import LibraryFilesCreator
from ms2query.ms2library import MS2Library
from ms2query.utils import save_pickled_file


def train_all_models(annotated_training_spectra,
                     unannotated_training_spectra,
                     output_folder):
    # set file names of new generated files
    ms2deepscore_model_file_name = os.path.join(output_folder, "ms2deepscore_model.hdf5")
    spec2vec_model_file_name = os.path.join(output_folder, "spec2vec_model.model")
    ms2query_model_file_name = os.path.join(output_folder, "ms2query_model.pickle")

    # Train MS2Deepscore model
    train_ms2deepscore_wrapper(annotated_training_spectra,
                               ms2deepscore_model_file_name)
    # Train Spec2Vec model
    train_spec2vec_model(annotated_training_spectra + unannotated_training_spectra,
                         os.path.join(spec2vec_model_file_name))

    ms2query_model = train_ms2query_model(
        annotated_training_spectra,
        os.path.join(output_folder, "library_files_for_training_ms2query"),
        ms2deepscore_model_file_name,
        spec2vec_model_file_name, 40)

    save_pickled_file(ms2query_model, ms2query_model_file_name)

    # Create embeddings and sqlite file for all annotated training spectra
    library_creator = LibraryFilesCreator(annotated_training_spectra,
                                          output_folder,
                                          s2v_model_file_name=spec2vec_model_file_name,
                                          ms2ds_model_file_name=ms2deepscore_model_file_name)
    library_creator.create_all_library_files()
