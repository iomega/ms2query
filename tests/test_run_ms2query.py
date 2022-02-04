import os
import sys
from ms2query.run_ms2query import download_default_models, run_complete_folder
from tests.test_ms2library import (MS2Library, create_test_classifier_csv_file,
                                   file_names, test_spectra)


if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


def test_download_default_models(tmp_path):
    """Tests downloading one of the files from zenodo

    Only one file is downloaded to keep test running time acceptable"""

    dir_to_store_files = os.path.join(tmp_path, "models")
    download_default_models(dir_to_store_files,
                            {"classifiers": "ALL_GNPS_210409_positive_processed_annotated_CF_NPC_classes.txt"})
    assert os.path.exists(dir_to_store_files)
    print(os.path.join(dir_to_store_files,
                                       "ALL_GNPS_210409_positive_processed_annotated_CF_NPC_classes.txt"))
    assert os.path.exists(os.path.join(dir_to_store_files,
                                       "ALL_GNPS_210409_positive_processed_annotated_CF_NPC_classes.txt"))
    with open(os.path.join(dir_to_store_files,
                           "ALL_GNPS_210409_positive_processed_annotated_CF_NPC_classes.txt"), "r") as classifiers_file:
        assert classifiers_file.readline() == "inchi_key\tsmiles\tcf_kingdom\tcf_superclass\tcf_class\t" \
                                              "cf_subclass\tcf_direct_parent\tnpc_class_results\t" \
                                              "npc_superclass_results\tnpc_pathway_results\tnpc_isglycoside\n"


def create_test_folder_with_spectra_files(path, spectra):
    """Creates a folder with two files containing two test spectra"""
    spectra_files_folder = os.path.join(path, "spectra_files_folder")
    os.mkdir(spectra_files_folder)

    pickle.dump(spectra, open(os.path.join(spectra_files_folder, "spectra_file_1.pickle"), "wb"))
    pickle.dump(spectra, open(os.path.join(spectra_files_folder, "spectra_file_2.pickle"), "wb"))
    return spectra_files_folder


def test_run_complete_folder(tmp_path, file_names, test_spectra):
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name, ms2q_model_file_name = file_names

    test_library = MS2Library(sqlite_file_loc, spec2vec_model_file_loc, ms2ds_model_file_name,
                              s2v_pickled_embeddings_file, ms2ds_embeddings_file_name, ms2q_model_file_name,
                              classifier_csv_file_name=None, spectrum_id_column_name=spectrum_id_column_name)

    folder_with_spectra = create_test_folder_with_spectra_files(tmp_path, test_spectra)
    results_directory = os.path.join(folder_with_spectra, "results")

    run_complete_folder(ms2library=test_library,
                        folder_with_spectra=folder_with_spectra,
                        minimal_ms2query_score=0)
    assert os.path.exists(results_directory), "Expected results directory to be created"

    assert os.listdir(os.path.join(results_directory)).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()

    with open(os.path.join(os.path.join(results_directory, 'spectra_file_1.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,retention_time,retention_index\n',
                '0,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,CCMSLIB00000001548,Hoiamide B,,\n',
                '1,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,CCMSLIB00000001570,Halovir A,,\n']

    with open(os.path.join(os.path.join(results_directory, 'spectra_file_2.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,retention_time,retention_index\n',
                '0,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,CCMSLIB00000001548,Hoiamide B,,\n',
                '1,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,CCMSLIB00000001570,Halovir A,,\n']


def test_run_complete_folder_with_classifiers(tmp_path, file_names, test_spectra):
    sqlite_file_loc, spec2vec_model_file_loc, s2v_pickled_embeddings_file, \
        ms2ds_model_file_name, ms2ds_embeddings_file_name, \
        spectrum_id_column_name, ms2q_model_file_name = file_names
    classifiers_file_location = create_test_classifier_csv_file(tmp_path)

    test_library = MS2Library(sqlite_file_loc, spec2vec_model_file_loc, ms2ds_model_file_name,
                              s2v_pickled_embeddings_file, ms2ds_embeddings_file_name, ms2q_model_file_name,
                              classifier_csv_file_name=classifiers_file_location,
                              spectrum_id_column_name=spectrum_id_column_name)

    folder_with_spectra = create_test_folder_with_spectra_files(tmp_path, test_spectra)
    results_directory = os.path.join(folder_with_spectra, "results")

    run_complete_folder(ms2library=test_library,
                        folder_with_spectra=folder_with_spectra,
                        minimal_ms2query_score=0,
                        additional_metadata_columns=["charge"],
                        additional_ms2query_score_columns=["s2v_score", "ms2ds_score"]
                        )
    assert os.path.exists(results_directory), "Expected results directory to be created"

    assert os.listdir(os.path.join(results_directory)).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()

    with open(os.path.join(os.path.join(results_directory, 'spectra_file_1.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,charge,s2v_score,ms2ds_score,smiles,cf_kingdom,cf_superclass,cf_class,cf_subclass,cf_direct_parent,npc_class_results,npc_superclass_results,npc_pathway_results\n',
                '0,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,CCMSLIB00000001548,Hoiamide B,1,0.9996,0.9232,CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O,Organic compounds,Organic acids and derivatives,Peptidomimetics,Depsipeptides,Cyclic depsipeptides,Cyclic peptides,Oligopeptides,Amino acids and Peptides\n',
                '1,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,CCMSLIB00000001570,Halovir A,0,0.9621,0.4600,nan,nan,nan,nan,nan,nan,nan,nan,nan\n']
    with open(os.path.join(os.path.join(results_directory, 'spectra_file_2.csv')), "r") as file:
        assert file.readlines() == \
               ['query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_ids,analog_compound_name,charge,s2v_score,ms2ds_score,smiles,cf_kingdom,cf_superclass,cf_class,cf_subclass,cf_direct_parent,npc_class_results,npc_superclass_results,npc_pathway_results\n',
                '0,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,CCMSLIB00000001548,Hoiamide B,1,0.9996,0.9232,CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O,Organic compounds,Organic acids and derivatives,Peptidomimetics,Depsipeptides,Cyclic depsipeptides,Cyclic peptides,Oligopeptides,Amino acids and Peptides\n',
                '1,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,CCMSLIB00000001570,Halovir A,0,0.9621,0.4600,nan,nan,nan,nan,nan,nan,nan,nan,nan\n']
