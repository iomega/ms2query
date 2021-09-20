import os
import pandas as pd
import numpy as np
from numpy import nan as Nan
from ms2query.results_table import get_classifier_from_csv_file, \
    add_classifiers_to_df


def test_get_classifier_from_csv_file(tmp_path):
    with open(os.path.join(tmp_path, "test_csv_file"), "w") as test_file:
        test_file.write(
            "inchi_key	smiles	cf_kingdom	cf_superclass	cf_class	cf_subclass	cf_direct_parent	npc_class_results	npc_superclass_results	npc_pathway_results	npc_isglycoside\n"
            "IYDKWWDUBYWQGF-NNAZGLEUSA-N	CC(C)CC1NC(=O)C(C)NC(=O)C(=C)N(C)C(=O)CCC(NC(=O)C(C)C(NC(=O)C(CCCNC(N)=N)NC(=O)C(C)C(NC1=O)C(O)=O)\C=C\C(\C)=C\C(C)C(O)Cc1ccccc1)C(O)=O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Hybrid peptides	Hybrid peptides	Cyclic peptides; Microcystins	Oligopeptides	Amino acids and Peptides	0\n"
            "KNGPFNUOXXLKCN-ZNCJFREWSA-N	CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Depsipeptides	Cyclic depsipeptides	Cyclic peptides	Oligopeptides	Amino acids and Peptides	0\n"
            "WXDBUBIFYCCNLE-NSCMQRKRSA-N	CCCCCCC[C@@H](C/C=C/CCC(=O)NC/C(=C/Cl)/[C@@]12[C@@H](O1)[C@H](CCC2=O)O)OC	Organic compounds	Organoheterocyclic compounds	Oxepanes		Oxepanes	Lipopeptides	Oligopeptides	Amino acids and Peptides	0")
    output = get_classifier_from_csv_file(
        os.path.join(tmp_path, "test_csv_file"),
        {"IYDKWWDUBYWQGF", "BBBBBBBBBBBBBB"})
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
            [["BBBBBBBBBBBBBB"] + [Nan] * 9]), columns=expected_columns)
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
        set())
    assert isinstance(output, pd.DataFrame), "Expected pandas DataFrame"
    expected_columns = ["inchikey", "smiles", "cf_kingdom", "cf_superclass",
                        "cf_class", "cf_subclass", "cf_direct_parent",
                        "npc_class_results", "npc_superclass_results",
                        "npc_pathway_results"]
    # Test if empty dataset with correct columns is created
    pd.testing.assert_frame_equal(output,
                                  pd.DataFrame(columns=expected_columns),
                                  check_dtype=False)
