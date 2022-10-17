import os
from typing import List
from matchms import Spectrum
import itertools
from ms2query.create_new_library.clean_and_split_data_for_training import split_spectra_in_random_inchikey_sets
from ms2query.create_new_library.train_ms2deepscore import train_ms2deepscore_wrapper
from ms2query.create_new_library.train_spec2vec import train_spec2vec_model
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
        train_all_models(training_set, unannotated_spectra, output_folder


def train_all_models(annotated_training_spectra,
                     unannotated_training_spectra,
                     output_folder):
    ms2deepscore_model_file_name = os.path.join(output_folder, "ms2deepscore_model.hdf5")
    spec2vec_model_file_name = os.path.join(output_folder, "spec2vec_model.model")
    ms2query_model_file_name = os.path.join(output_folder, "ms2query_model.pickle")

    # Train MS2Deepscore model
    train_ms2deepscore_wrapper(annotated_training_spectra,
                               ms2deepscore_model_file_name)
    # Train Spec2Vec model
    train_spec2vec_model(annotated_training_spectra + unannotated_training_spectra,
                         os.path.join(spec2vec_model_file_name))
    # Create library files
    # library_creator = LibraryFilesCreator(library_spectra,
    #                                       output_directory=output_folder,
    #                                       ion_mode="positive",
    #                                       ms2ds_model_file_name=,
    #                                       s2v_model_file_name=, )
    # library_creator.clean_up_smiles_inchi_and_inchikeys(do_pubchem_lookup=True)
    # library_creator.clean_peaks_and_normalise_intensities_spectra()
    # library_creator.remove_not_fully_annotated_spectra()
    # library_creator.create_all_library_files()

    # Create training data MS2Query model
    # Train MS2Query model






if __name__ == "__main__":
    pass
    # Train MS2Deepscore and Spec2Vec on 4/5th training spectra
    # Train MS2Query with MS2Deepscore and Spec2Vec
    # Use test set on MS2Query to test performance
    # Store the complete model and the 5 data splits.
