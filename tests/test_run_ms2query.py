import os
import pickle
import pytest
from ms2query.run_ms2query import download_default_models, run_complete_folder
from tests.test_ms2library import file_names, MS2Library, create_test_classifier_csv_file, test_spectra


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
    classifiers_file_location = create_test_classifier_csv_file(tmp_path)

    test_library = MS2Library(sqlite_file_loc, spec2vec_model_file_loc, ms2ds_model_file_name,
                              s2v_pickled_embeddings_file, ms2ds_embeddings_file_name, ms2q_model_file_name,
                              classifier_csv_file_name=None, spectrum_id_column_name=spectrum_id_column_name)

    results_directory = os.path.join(tmp_path, "results")
    run_complete_folder(ms2library=test_library,
                        folder_with_spectra=create_test_folder_with_spectra_files(tmp_path, test_spectra),
                        results_folder=results_directory,
                        minimal_ms2query_score=0)
    assert os.path.exists(results_directory), "Expected results directory to be created"
    assert os.path.exists(os.path.join(results_directory, "analog_search")), \
        "Expected analog search directory to be created"
    assert os.path.exists(os.path.join(results_directory, "library_search")), \
        "Expected library search directory to be created"
    assert os.listdir(os.path.join(results_directory, "analog_search")).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()
    assert os.listdir(os.path.join(results_directory, "library_search")).sort() == ['spectra_file_1.csv', 'spectra_file_2.csv'].sort()

    with open(os.path.join(os.path.join(results_directory, "analog_search", 'spectra_file_1.csv')), "r") as file:
        assert file.readlines() == [',parent_mass_query_spectrum,ms2query_model_prediction,parent_mass_analog,inchikey,spectrum_ids,analog_compound_name\n', '0,905.9927235480093,0.5706255,466.200724,HKSZLNNOFSGOKW,CCMSLIB00000001655,Staurosporine\n', '0,926.9927235480093,0.5717702,736.240782,HEWGADDUUGVTPF,CCMSLIB00000001640,Antanapeptin A\n']
    with open(os.path.join(os.path.join(results_directory, "analog_search", 'spectra_file_2.csv')), "r") as file:
        assert file.readlines() == [',parent_mass_query_spectrum,ms2query_model_prediction,parent_mass_analog,inchikey,spectrum_ids,analog_compound_name\n', '0,905.9927235480093,0.5706255,466.200724,HKSZLNNOFSGOKW,CCMSLIB00000001655,Staurosporine\n', '0,926.9927235480093,0.5717702,736.240782,HEWGADDUUGVTPF,CCMSLIB00000001640,Antanapeptin A\n']

    with open(os.path.join(os.path.join(results_directory, "library_search", 'spectra_file_1.csv')), "r") as file:
        assert file.readlines() == ['query_spectrum_nr,query_spectrum_parent_mass,s2v_score,match_spectrum_id,match_parent_mass,match_inchikey,match_compound_name\n']
    with open(os.path.join(os.path.join(results_directory, "library_search", 'spectra_file_2.csv')), "r") as file:
        assert file.readlines() == ['query_spectrum_nr,query_spectrum_parent_mass,s2v_score,match_spectrum_id,match_parent_mass,match_inchikey,match_compound_name\n']

