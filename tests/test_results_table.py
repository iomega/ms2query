

def test_get_classifier_from_csv_file(tmp_path):
    test_csv_file = """inchi_key	smiles	cf_kingdom	cf_superclass	cf_class	cf_subclass	cf_direct_parent	npc_class_results	npc_superclass_results	npc_pathway_results	npc_isglycoside
                IYDKWWDUBYWQGF-NNAZGLEUSA-N	CC(C)CC1NC(=O)C(C)NC(=O)C(=C)N(C)C(=O)CCC(NC(=O)C(C)C(NC(=O)C(CCCNC(N)=N)NC(=O)C(C)C(NC1=O)C(O)=O)\C=C\C(\C)=C\C(C)C(O)Cc1ccccc1)C(O)=O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Hybrid peptides	Hybrid peptides	Cyclic peptides; Microcystins	Oligopeptides	Amino acids and Peptides	0
                KNGPFNUOXXLKCN-ZNCJFREWSA-N	CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O	Organic compounds	Organic acids and derivatives	Peptidomimetics	Depsipeptides	Cyclic depsipeptides	Cyclic peptides	Oligopeptides	Amino acids and Peptides	0
                WXDBUBIFYCCNLE-NSCMQRKRSA-N	CCCCCCC[C@@H](C/C=C/CCC(=O)NC/C(=C/Cl)/[C@@]12[C@@H](O1)[C@H](CCC2=O)O)OC	Organic compounds	Organoheterocyclic compounds	Oxepanes		Oxepanes	Lipopeptides	Oligopeptides	Amino acids and Peptides	0
                """
