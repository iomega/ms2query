import os
import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from ms2query.utils import (add_unknown_charges_to_spectra,
                            load_matchms_spectrum_objects_from_file,
                            get_classifier_from_csv_file, load_pickled_file)


@pytest.fixture(scope="package")
def path_to_general_test_files() -> str:
    return os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files')


def test_convert_files_to_matchms_spectrum_objects_unknown_file(tmp_path):
    """Tests if unknown file raises an Assertion error"""
    with pytest.raises(AssertionError):
        load_matchms_spectrum_objects_from_file(os.path.join(tmp_path, "file_that_does_not_exist.json"))


def test_convert_files_to_matchms_spectrum_object_known_file():
    """Test if pickled file is loaded in correctly"""
    spectra = load_matchms_spectrum_objects_from_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files/100_test_spectra.pickle'))
    assert isinstance(spectra, list), "expected list of spectra"
    assert len(spectra) == 100, "expected 100 spectra"
    for spectrum in spectra:
        assert isinstance(spectrum, Spectrum)


def test_add_unknown_charges_to_spectra():
    spectra = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files/100_test_spectra.pickle'))
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


def create_test_classifier_csv_file(tmp_path):
    file_location = os.path.join(tmp_path, "test_csv_file")
    with open(file_location, "w") as test_file:
        test_file.write(
            "inchi_key	smiles	cf_kingdom	cf_superclass	cf_class	cf_subclass	cf_direct_parent	npc_class_results	npc_superclass_results	npc_pathway_results	npc_isglycoside\n"
            "IYDKWWDUBYWQGF-NNAZGLEUSA-N	CC(C)CC1NC(=O)C(C)NC(=O)C(=C)N(C)C(=O)CCC(NC(=O)C(C)C(NC(=O)C(CCCNC(N)=N)NC(=O)C(C)C(NC1=O)C(O)=O)\C=C\C(\C)=C\C(C)C(O)Cc1ccccc1)C(O)=O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Hybrid peptides	Hybrid peptides	Cyclic peptides; Microcystins	Oligopeptides	Amino acids and Peptides	0\n"
            "KNGPFNUOXXLKCN-ZNCJFREWSA-N	CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Depsipeptides	Cyclic depsipeptides	Cyclic peptides	Oligopeptides	Amino acids and Peptides	0\n"
            "HKSZLNNOFSGOKW-NSCMQRKRSA-N	CCCCCCC[C@@H](C/C=C/CCC(=O)NC/C(=C/Cl)/[C@@]12[C@@H](O1)[C@H](CCC2=O)O)OC	Organic compounds	Organoheterocyclic compounds	Oxepanes		Oxepanes	Lipopeptides	Oligopeptides	Amino acids and Peptides	0")
    return file_location


def test_get_classifier_from_csv_file(tmp_path):
    create_test_classifier_csv_file(tmp_path)
    output = get_classifier_from_csv_file(
        os.path.join(tmp_path, "test_csv_file"),
        ["IYDKWWDUBYWQGF", "BBBBBBBBBBBBBB"])
    assert isinstance(output, pd.DataFrame), "Expected pandas DataFrame"
    expected_columns = ["inchikey", "smiles", "cf_kingdom", "cf_superclass",
                        "cf_class", "cf_subclass", "cf_direct_parent",
                        "npc_class_results", "npc_superclass_results",
                        "npc_pathway_results"]
    assert all([column_name in output.columns
                for column_name in expected_columns]), \
        "Expected different column names"
    assert output.shape == (2, 10), "Expected different shape of dataframe"
    assert "IYDKWWDUBYWQGF" in list(output["inchikey"]), \
        "IYDKWWDUBYWQGF was expected in column inchikey"
    assert "BBBBBBBBBBBBBB" in list(output["inchikey"]), \
        "BBBBBBBBBBBBBB was expected in column inchikey"
    pd.testing.assert_frame_equal(
        output[output["inchikey"] == "BBBBBBBBBBBBBB"].reset_index(drop=True),
        pd.DataFrame(np.array(
            [["BBBBBBBBBBBBBB"] + [np.nan] * 9]), columns=expected_columns)
    )
    pd.testing.assert_frame_equal(
        output[output["inchikey"] == "IYDKWWDUBYWQGF"].reset_index(drop=True),
        pd.DataFrame(np.array(["IYDKWWDUBYWQGF",
                               "CC(C)CC1NC(=O)C(C)NC(=O)C(=C)N(C)C(=O)CCC(NC(=O)C(C)C(NC(=O)C(CCCNC(N)=N)NC(=O)C(C)C(NC1=O)C(O)=O)\C=C\C(\C)=C\C(C)C(O)Cc1ccccc1)C(O)=O",
                               "Organic compounds",
                               "Organic acids and derivatives",
                               "Peptidomimetics", "Hybrid peptides",
                               "Hybrid peptides",
                               "Cyclic peptides; Microcystins",
                               "Oligopeptides",
                               "Amino acids and Peptides"]).reshape(-1, 10),
                     columns=expected_columns)
    )


def test_get_classifier_from_csv_file_empty(tmp_path):
    """Test if empty dataframe is returned, when no inchikeys are given"""
    create_test_classifier_csv_file(tmp_path)
    output = get_classifier_from_csv_file(
        os.path.join(tmp_path, "test_csv_file"),
        [])
    assert isinstance(output, pd.DataFrame), "Expected pandas DataFrame"
    expected_columns = ["inchikey", "smiles", "cf_kingdom", "cf_superclass",
                        "cf_class", "cf_subclass", "cf_direct_parent",
                        "npc_class_results", "npc_superclass_results",
                        "npc_pathway_results"]
    # Test if empty dataset with correct columns is created
    pd.testing.assert_frame_equal(output,
                                  pd.DataFrame(columns=expected_columns),
                                  check_dtype=False)
