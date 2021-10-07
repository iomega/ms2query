import os
import numpy as np
import pytest
import pandas as pd
from matchms import Spectrum
from ms2query import ResultsTable
from ms2query.results_table import get_classifier_from_csv_file


@pytest.fixture
def dummy_data():
    ms2deepscores = pd.DataFrame(np.array([0.2, 0.7, 0.99, 0.4]),
                                 index=["XXXXXXXXXXXXXA",
                                          "XXXXXXXXXXXXXB",
                                          "XXXXXXXXXXXXXC",
                                          "XXXXXXXXXXXXXD"])

    query_spectrum = Spectrum(mz=np.array([100.0]),
                              intensities=np.array([1.0]),
                              metadata={"parent_mass": 205.0})
    sqlite_test_file = "test_files/general_test_files/100_test_spectra.sqlite"
    return ms2deepscores, query_spectrum, sqlite_test_file


def test_table_init(dummy_data):
    ms2deepscores, query_spectrum, sqlite_test_file = dummy_data
    preselection_cut_off = 3
    table = ResultsTable(preselection_cut_off,
                         ms2deepscores.iloc[:, 0],
                         query_spectrum,
                         sqlite_test_file)
    assert table.data.shape == (0, 11), \
        "Should have an empty data attribute"
    assert table.parent_mass == 205.0, \
        "Expected different parent mass"


def test_table_preselect_ms2deepscore(dummy_data):
    ms2deepscores, query_spectrum, sqlite_test_file = dummy_data
    preselection_cut_off = 3
    table = ResultsTable(preselection_cut_off,
                         ms2deepscores.iloc[:, 0],
                         query_spectrum,
                         sqlite_test_file)
    table.preselect_on_ms2deepscore()
    assert table.data.shape == (3, 11), "Should have different data table"
    assert np.all(table.data.spectrum_ids.values == \
                  ['XXXXXXXXXXXXXC', 'XXXXXXXXXXXXXB', 'XXXXXXXXXXXXXD']), \
        "Expected different spectrum IDs or order"
    assert np.all(table.data.ms2ds_score.values == \
                  [0.99, 0.7, 0.4]), \
        "Expected different scores or order"


def test_add_parent_masses(dummy_data):
    ms2deepscores, query_spectrum, sqlite_test_file = dummy_data
    preselection_cut_off = 2
    table = ResultsTable(preselection_cut_off,
                         ms2deepscores.iloc[:, 0],
                         query_spectrum,
                         sqlite_test_file)
    table.add_parent_masses(np.array([190.0, 199.2, 200.0, 201.0]), 0.8)
    expected = np.array([0.03518437, 0.27410813, 0.32768, 0.4096])

    assert table.data.shape == (4, 11), \
        "Should have different data table"
    assert np.all(np.isclose(table.data.mass_similarity.values,
                             expected, atol=1e-7)), \
        "Expected different scores or order"

    
def test_get_classifier_from_csv_file(tmp_path):
    with open(os.path.join(tmp_path, "test_csv_file"), "w") as test_file:
        test_file.write(
            "inchi_key	smiles	cf_kingdom	cf_superclass	cf_class	cf_subclass	cf_direct_parent	npc_class_results	npc_superclass_results	npc_pathway_results	npc_isglycoside\n"
            "IYDKWWDUBYWQGF-NNAZGLEUSA-N	CC(C)CC1NC(=O)C(C)NC(=O)C(=C)N(C)C(=O)CCC(NC(=O)C(C)C(NC(=O)C(CCCNC(N)=N)NC(=O)C(C)C(NC1=O)C(O)=O)\C=C\C(\C)=C\C(C)C(O)Cc1ccccc1)C(O)=O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Hybrid peptides	Hybrid peptides	Cyclic peptides; Microcystins	Oligopeptides	Amino acids and Peptides	0\n"
            "KNGPFNUOXXLKCN-ZNCJFREWSA-N	CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Depsipeptides	Cyclic depsipeptides	Cyclic peptides	Oligopeptides	Amino acids and Peptides	0\n"
            "WXDBUBIFYCCNLE-NSCMQRKRSA-N	CCCCCCC[C@@H](C/C=C/CCC(=O)NC/C(=C/Cl)/[C@@]12[C@@H](O1)[C@H](CCC2=O)O)OC	Organic compounds	Organoheterocyclic compounds	Oxepanes		Oxepanes	Lipopeptides	Oligopeptides	Amino acids and Peptides	0")
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
    with open(os.path.join(tmp_path, "test_csv_file"), "w") as test_file:
        test_file.write(
            "inchi_key	smiles	cf_kingdom	cf_superclass	cf_class	cf_subclass	cf_direct_parent	npc_class_results	npc_superclass_results	npc_pathway_results	npc_isglycoside\n"
            "IYDKWWDUBYWQGF-NNAZGLEUSA-N	CC(C)CC1NC(=O)C(C)NC(=O)C(=C)N(C)C(=O)CCC(NC(=O)C(C)C(NC(=O)C(CCCNC(N)=N)NC(=O)C(C)C(NC1=O)C(O)=O)\C=C\C(\C)=C\C(C)C(O)Cc1ccccc1)C(O)=O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Hybrid peptides	Hybrid peptides	Cyclic peptides; Microcystins	Oligopeptides	Amino acids and Peptides	0\n"
            "KNGPFNUOXXLKCN-ZNCJFREWSA-N	CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Depsipeptides	Cyclic depsipeptides	Cyclic peptides	Oligopeptides	Amino acids and Peptides	0\n"
            "WXDBUBIFYCCNLE-NSCMQRKRSA-N	CCCCCCC[C@@H](C/C=C/CCC(=O)NC/C(=C/Cl)/[C@@]12[C@@H](O1)[C@H](CCC2=O)O)OC	Organic compounds	Organoheterocyclic compounds	Oxepanes		Oxepanes	Lipopeptides	Oligopeptides	Amino acids and Peptides	0")
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
