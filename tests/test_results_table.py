import os
import numpy as np
import pytest
import pandas as pd
from matchms import Spectrum
from ms2query import ResultsTable
from ms2query.utils import load_pickled_file
from tests.test_utils import create_test_classifier_csv_file


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
    assert np.all(table.data.spectrum_ids.values ==
                  ['XXXXXXXXXXXXXC', 'XXXXXXXXXXXXXB', 'XXXXXXXXXXXXXD']), \
        "Expected different spectrum IDs or order"
    assert np.all(table.data.ms2ds_score.values ==
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


def test_export_to_dataframe(dummy_data, tmp_path):
    create_test_classifier_csv_file(tmp_path)
    test_table: ResultsTable = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_analog_search_results.pickle'))[0]
    test_table.sqlite_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0], "tests/test_files/general_test_files/100_test_spectra.sqlite")
    test_table.classifier_csv_file_name = os.path.join(tmp_path, "test_csv_file")
    returned_dataframe = test_table.export_to_dataframe(5)
    assert isinstance(returned_dataframe, pd.DataFrame)
    assert list(returned_dataframe.columns) == ['parent_mass_query_spectrum', 'ms2query_model_prediction',
       'parent_mass_analog', 'inchikey', 'spectrum_ids',
       'analog_compound_name', 'smiles', 'cf_kingdom', 'cf_superclass',
       'cf_class', 'cf_subclass', 'cf_direct_parent', 'npc_class_results',
       'npc_superclass_results', 'npc_pathway_results']
    # Check if one of the classifiers is filled in
    assert returned_dataframe["npc_pathway_results"][0] == "Amino acids and Peptides"
    assert len(returned_dataframe.index) == 5
    # Test if first row is correct
    np.testing.assert_array_almost_equal(
        list(returned_dataframe.iloc[0, :3]),
        [905.9927235480093, 0.5706255, 466.200724],
        5)
    assert np.all(list(returned_dataframe.iloc[0, 3:10]) ==
                       ['HKSZLNNOFSGOKW', 'CCMSLIB00000001655', 'Staurosporine', 'CCCCCCC[C@@H](C/C=C/CCC(=O)NC/C(=C/Cl)/[C@@]12[C@@H](O1)[C@H](CCC2=O)O)OC', 'Organic compounds', 'Organoheterocyclic compounds', 'Oxepanes'])
