import os
import sys
from ms2query.ms2library import create_library_object_from_one_dir
from ms2query.run_ms2query import download_zenodo_files, run_complete_folder
from ms2query.utils import SettingsRunMS2Query
from tests.test_ms2library import (MS2Library,
                                   ms2library, test_spectra)
from tests.test_utils import create_test_classifier_csv_file

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


def test_download_default_models(tmp_path):
    """Tests downloading small files from zenodo

    The files are a total of 20 MB from https://zenodo.org/record/7108049#.Yy2nPKRBxPY"""
    dir_to_store_test_files = os.path.join(tmp_path, "positive_model")
    download_zenodo_files(7108049, dir_to_store_test_files)
    assert os.path.exists(dir_to_store_test_files)
    assert os.path.exists(os.path.join(dir_to_store_test_files,
                                       "GNPS_15_12_2021_neg_test_250_inchikeys.pickle"))
    assert os.path.exists(os.path.join(dir_to_store_test_files,
                                       "GNPS_15_12_2021_neg_test_3000_spectra.pickle"))
    assert os.path.exists(os.path.join(dir_to_store_test_files,
                                       "GNPS_15_12_2021_pos_test_250_inchikeys.pickle"))
    assert os.path.exists(os.path.join(dir_to_store_test_files,
                                       "GNPS_15_12_2021_pos_test_3000_spectra.pickle"))

    run_test = False # Run test is set to false, since downloading takes too long for default testing
    if run_test:
        dir_to_store_positive_files = os.path.join(tmp_path, "positive_model")
        dir_to_store_negative_files = os.path.join(tmp_path, "negative_model")

        download_zenodo_files(6997924, dir_to_store_positive_files)
        download_zenodo_files(7107654, dir_to_store_negative_files)
        assert os.path.exists(dir_to_store_positive_files)
        assert os.path.exists(dir_to_store_negative_files)
        pos_ms2library = create_library_object_from_one_dir(dir_to_store_positive_files)
        neg_ms2library = create_library_object_from_one_dir(dir_to_store_negative_files)
        assert isinstance(pos_ms2library, MS2Library)
        assert isinstance(neg_ms2library, MS2Library)


def create_test_folder_with_spectra_files(path, spectra):
    """Creates a folder with two files containing two test spectra"""
    spectra_files_folder = os.path.join(path, "spectra_files_folder")
    os.mkdir(spectra_files_folder)

    pickle.dump(spectra, open(os.path.join(spectra_files_folder, "spectra_file_1.pickle"), "wb"))
    pickle.dump(spectra, open(os.path.join(spectra_files_folder, "spectra_file_2.pickle"), "wb"))
    return spectra_files_folder


def test_run_complete_folder(tmp_path, ms2library, test_spectra):
    folder_with_spectra = create_test_folder_with_spectra_files(tmp_path, test_spectra)
    results_directory = os.path.join(folder_with_spectra, "results")

    run_complete_folder(ms2library=ms2library,
                        folder_with_spectra=folder_with_spectra)
    assert os.path.exists(results_directory), "Expected results directory to be created"

    assert os.listdir(os.path.join(results_directory)).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()

    with open(os.path.join(os.path.join(results_directory, 'spectra_file_1.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,retention_time,retention_index\n',
                '1,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,CCMSLIB00000001548,Hoiamide B,,\n',
                '2,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,CCMSLIB00000001570,Halovir A,,\n']

    with open(os.path.join(os.path.join(results_directory, 'spectra_file_2.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,retention_time,retention_index\n',
                '1,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,CCMSLIB00000001548,Hoiamide B,,\n',
                '2,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,CCMSLIB00000001570,Halovir A,,\n']


def test_run_complete_folder_with_classifiers(tmp_path, ms2library, test_spectra):
    classifiers_file_location = create_test_classifier_csv_file(tmp_path)

    ms2library.classifier_file_name = classifiers_file_location
    folder_with_spectra = create_test_folder_with_spectra_files(tmp_path, test_spectra)
    results_directory = os.path.join(folder_with_spectra, "results")
    settings = SettingsRunMS2Query(minimal_ms2query_metascore=0,
                                   additional_metadata_columns=("charge",),
                                   additional_ms2query_score_columns=("s2v_score", "ms2ds_score"))
    run_complete_folder(ms2library=ms2library,
                        folder_with_spectra=folder_with_spectra,
                        settings=settings
                        )
    assert os.path.exists(results_directory), "Expected results directory to be created"

    assert os.listdir(os.path.join(results_directory)).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()

    with open(os.path.join(os.path.join(results_directory, 'spectra_file_1.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,charge,s2v_score,ms2ds_score,smiles,cf_kingdom,cf_superclass,cf_class,cf_subclass,cf_direct_parent,npc_class_results,npc_superclass_results,npc_pathway_results\n',
                '1,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,CCMSLIB00000001548,Hoiamide B,1,0.9996,0.9232,CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O,Organic compounds,Organic acids and derivatives,Peptidomimetics,Depsipeptides,Cyclic depsipeptides,Cyclic peptides,Oligopeptides,Amino acids and Peptides\n',
                '2,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,CCMSLIB00000001570,Halovir A,0,0.9621,0.4600,nan,nan,nan,nan,nan,nan,nan,nan,nan\n']
    with open(os.path.join(os.path.join(results_directory, 'spectra_file_2.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,charge,s2v_score,ms2ds_score,smiles,cf_kingdom,cf_superclass,cf_class,cf_subclass,cf_direct_parent,npc_class_results,npc_superclass_results,npc_pathway_results\n',
                '1,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,CCMSLIB00000001548,Hoiamide B,1,0.9996,0.9232,CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O,Organic compounds,Organic acids and derivatives,Peptidomimetics,Depsipeptides,Cyclic depsipeptides,Cyclic peptides,Oligopeptides,Amino acids and Peptides\n',
                '2,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,CCMSLIB00000001570,Halovir A,0,0.9621,0.4600,nan,nan,nan,nan,nan,nan,nan,nan,nan\n']
