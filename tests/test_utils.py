import os
from io import StringIO
from typing import List

import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from ms2query.utils import (add_unknown_charges_to_spectra,
                            load_matchms_spectrum_objects_from_file,
                            load_pickled_file,
                            convert_to_onnx_model, load_ms2query_model, predict_onnx_model)


def test_convert_files_to_matchms_spectrum_objects_unknown_file(tmp_path):
    """Tests if unknown file raises an Assertion error"""
    with pytest.raises(AssertionError):
        load_matchms_spectrum_objects_from_file(os.path.join(tmp_path, "file_that_does_not_exist.json"))


def test_convert_files_to_matchms_spectrum_object_known_file():
    """Test if pickled file is loaded in correctly"""
    spectra = load_matchms_spectrum_objects_from_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files/100_test_spectra.mgf'))
    assert isinstance(spectra, list), "expected list of spectra"
    assert len(spectra) == 100, "expected 100 spectra"
    for spectrum in spectra:
        assert isinstance(spectrum, Spectrum)


def test_add_unknown_charges_to_spectra(hundred_test_spectra):
    spectra = hundred_test_spectra
    # Set charges to predefined values
    for spectrum in spectra[:10]:
        spectrum.set("charge", None)
    for spectrum in spectra[10:20]:
        spectrum.set("charge", 1)
    for spectrum in spectra[20:30]:
        spectrum.set("charge", -1)
    for spectrum in spectra[30:]:
        spectrum.set("charge", 2)

    spectra_with_charge = add_unknown_charges_to_spectra(spectra)
    # Test if charges are set correctly
    for spectrum in spectra_with_charge[:20]:
        assert spectrum.get("charge") == 1, "The charge is expected to be 1"
    for spectrum in spectra_with_charge[20:30]:
        assert spectrum.get("charge") == -1, "The charge is expected to be -1"
    for spectrum in spectra_with_charge[30:]:
        assert spectrum.get("charge") == 2, "The charge is expected to be 2"


def test_save_as_onnx_model(tmp_path):
    path_to_test_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/')
    rf_model_file = os.path.join(path_to_test_dir, 'general_test_files', "test_ms2q_rf_model.pickle")
    rf_model = load_pickled_file(rf_model_file)
    expected_result = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "tests/test_files/test_files_train_ms2query_nn",
        "expected_train_and_val_data.pickle"))[0]
    new_model = os.path.join(tmp_path, "rf_model.onnx")
    convert_to_onnx_model(rf_model, new_model)
    ms2query_model = load_ms2query_model(new_model)
    result = predict_onnx_model(ms2query_model, expected_result.values)
    original_result = rf_model.predict(expected_result.values.astype(np.float32))
    assert np.allclose(result, original_result)


def check_correct_results_csv_file(dataframe_found: pd.DataFrame,
                                   expected_headers: List[str],
                                   nr_of_rows_to_check=2):
    # Define expected results
    csv_format_expected_results ="""query_spectrum_nr,ms2query_model_prediction,precursor_mz_difference,precursor_mz_query_spectrum,precursor_mz_analog,inchikey,spectrum_id,analog_compound_name,charge,s2v_score,ms2ds_score,retention_time,retention_index,smiles,cf_kingdom,cf_superclass,cf_class,cf_subclass,cf_direct_parent,npc_class_results,npc_superclass_results,npc_pathway_results\n
        1,0.5645,33.2500,907.0000,940.2500,KNGPFNUOXXLKCN,CCMSLIB00000001760,Hoiamide B,1,0.9996,0.9232,,,CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O,b,c,d,e,f,g,h,i\n
        2,0.4090,61.3670,928.0000,866.6330,GRJSOZDXIUZXEW,CCMSLIB00000001761,Halovir A,0,0.9621,0.4600,,,CCCCCCCCCCCCCC(=O)NC(C)(C)C(=O)N1C[C@H](O)C[C@H]1C(=O)NC(CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@H](CO)CC(C)C,b,c,d,e,f,g,h,i\n"""
    dataframe_expected_results = pd.read_csv(StringIO(csv_format_expected_results), sep=",", header=0)

    # convert csv rows to dataframe
    check_expected_headers(dataframe_found, expected_headers)

    # Select only the matching columns
    selection_of_matching_headers = dataframe_expected_results[dataframe_found.columns]
    pd.testing.assert_frame_equal(dataframe_found.iloc[:nr_of_rows_to_check, :],
                                  selection_of_matching_headers.iloc[:nr_of_rows_to_check, :],
                                  check_dtype=False,
                                  rtol=1.0e-4)


def check_expected_headers(dataframe_found: pd.DataFrame,
                           expected_headers: List[str]):
    found_headers = list(dataframe_found.columns)
    assert len(found_headers) == len(found_headers)
    for i, header in enumerate(expected_headers):
        assert header == found_headers[i]