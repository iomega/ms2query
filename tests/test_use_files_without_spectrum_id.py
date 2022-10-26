import os
import sys
import pytest
from ms2query.run_ms2query import run_complete_folder
from tests.test_ms2library import (MS2Library,
                                   test_spectra)
from tests.test_utils import create_test_classifier_csv_file


if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


@pytest.fixture
def ms2library_without_spectrum_id():
    """Returns file names of the files needed to create MS2Library object"""
    path_to_tests_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/')
    sqlite_file_loc = os.path.join(
        path_to_tests_dir,
        "general_test_files", "test_files_without_spectrum_id", "100_test_spectra.sqlite")
    spec2vec_model_file_loc = os.path.join(
        path_to_tests_dir,
        "general_test_files/100_test_spectra_s2v_model.model")
    s2v_pickled_embeddings_file = os.path.join(
        path_to_tests_dir,
        "general_test_files", "test_files_without_spectrum_id", "100_test_spectra_s2v_embeddings.pickle")
    ms2ds_model_file_name = os.path.join(
        path_to_tests_dir,
        "general_test_files/ms2ds_siamese_210301_5000_500_400.hdf5")
    ms2ds_embeddings_file_name = os.path.join(
        path_to_tests_dir,
        "general_test_files", "test_files_without_spectrum_id", "100_test_spectra_ms2ds_embeddings.pickle")
    spectrum_id_column_name = "spectrumid"
    ms2q_model_file_name = os.path.join(path_to_tests_dir,
        "general_test_files", "test_ms2q_rf_model.pickle")
    ms2library = MS2Library(sqlite_file_loc,
                            spec2vec_model_file_loc,
                            ms2ds_model_file_name,
                            s2v_pickled_embeddings_file,
                            ms2ds_embeddings_file_name,
                            ms2q_model_file_name,
                            spectrum_id_column_name=spectrum_id_column_name)
    return ms2library


def create_test_folder_with_spectra_files(path, spectra):
    """Creates a folder with two files containing two test spectra"""
    spectra_files_folder = os.path.join(path, "spectra_files_folder")
    os.mkdir(spectra_files_folder)

    pickle.dump(spectra, open(os.path.join(spectra_files_folder, "spectra_file_1.pickle"), "wb"))
    pickle.dump(spectra, open(os.path.join(spectra_files_folder, "spectra_file_2.pickle"), "wb"))
    return spectra_files_folder


def test_run_complete_folder(tmp_path, ms2library_without_spectrum_id, test_spectra):
    folder_with_spectra = create_test_folder_with_spectra_files(tmp_path, test_spectra)
    results_directory = os.path.join(folder_with_spectra, "results")

    run_complete_folder(ms2library=ms2library_without_spectrum_id,
                        folder_with_spectra=folder_with_spectra,
                        minimal_ms2query_score=0)
    assert os.path.exists(results_directory), "Expected results directory to be created"

    assert os.listdir(os.path.join(results_directory)).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()

    with open(os.path.join(os.path.join(results_directory, 'spectra_file_1.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,retention_time,retention_index\n',
                '1,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,1,Hoiamide B,,\n',
                '2,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,16,Halovir A,,\n']

    with open(os.path.join(os.path.join(results_directory, 'spectra_file_2.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,retention_time,retention_index\n',
                '1,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,1,Hoiamide B,,\n',
                '2,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,16,Halovir A,,\n']


def test_run_complete_folder_with_classifiers(tmp_path, ms2library_without_spectrum_id, test_spectra):
    classifiers_file_location = create_test_classifier_csv_file(tmp_path)
    ms2library_without_spectrum_id.classifier_file_name = classifiers_file_location

    folder_with_spectra = create_test_folder_with_spectra_files(tmp_path, test_spectra)
    results_directory = os.path.join(folder_with_spectra, "results")

    run_complete_folder(ms2library=ms2library_without_spectrum_id,
                        folder_with_spectra=folder_with_spectra,
                        minimal_ms2query_score=0,
                        additional_metadata_columns=("charge",),
                        additional_ms2query_score_columns=["s2v_score", "ms2ds_score"]
                        )
    assert os.path.exists(results_directory), "Expected results directory to be created"

    assert os.listdir(os.path.join(results_directory)).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()

    with open(os.path.join(os.path.join(results_directory, 'spectra_file_1.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,charge,s2v_score,ms2ds_score,smiles,cf_kingdom,cf_superclass,cf_class,cf_subclass,cf_direct_parent,npc_class_results,npc_superclass_results,npc_pathway_results\n',
                '1,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,1,Hoiamide B,1,0.9996,0.9232,CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O,Organic compounds,Organic acids and derivatives,Peptidomimetics,Depsipeptides,Cyclic depsipeptides,Cyclic peptides,Oligopeptides,Amino acids and Peptides\n',
                '2,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,16,Halovir A,0,0.9621,0.4600,nan,nan,nan,nan,nan,nan,nan,nan,nan\n']
    with open(os.path.join(os.path.join(results_directory, 'spectra_file_2.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,charge,s2v_score,ms2ds_score,smiles,cf_kingdom,cf_superclass,cf_class,cf_subclass,cf_direct_parent,npc_class_results,npc_superclass_results,npc_pathway_results\n',
                '1,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,1,Hoiamide B,1,0.9996,0.9232,CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O,Organic compounds,Organic acids and derivatives,Peptidomimetics,Depsipeptides,Cyclic depsipeptides,Cyclic peptides,Oligopeptides,Amino acids and Peptides\n',
                '2,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,16,Halovir A,0,0.9621,0.4600,nan,nan,nan,nan,nan,nan,nan,nan,nan\n']
