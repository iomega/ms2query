import os
import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from ms2query import ResultsTable
from ms2query.utils import column_names_for_output, load_pickled_file
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
                              metadata={"precursor_mz": 205.0, "spectrum_nr": 0})
    sqlite_test_file = "test_files/general_test_files/100_test_spectra.sqlite"
    return ms2deepscores, query_spectrum, sqlite_test_file


def test_table_init(dummy_data):
    ms2deepscores, query_spectrum, sqlite_test_file = dummy_data
    preselection_cut_off = 3
    table = ResultsTable(preselection_cut_off,
                         ms2deepscores.iloc[:, 0],
                         query_spectrum,
                         sqlite_test_file)
    assert table.data.shape == (0, 8), \
        "Should have an empty data attribute"
    assert table.precursor_mz == 205.0, \
        "Expected different precursor m/z"


def test_table_preselect_ms2deepscore(dummy_data):
    ms2deepscores, query_spectrum, sqlite_test_file = dummy_data
    preselection_cut_off = 3
    table = ResultsTable(preselection_cut_off,
                         ms2deepscores.iloc[:, 0],
                         query_spectrum,
                         sqlite_test_file)
    table.preselect_on_ms2deepscore()
    assert table.data.shape == (3, 8), "Should have different data table"
    assert np.all(table.data.spectrum_ids.values ==
                  ['XXXXXXXXXXXXXC', 'XXXXXXXXXXXXXB', 'XXXXXXXXXXXXXD']), \
        "Expected different spectrum IDs or order"
    assert np.all(table.data.ms2ds_score.values ==
                  [0.99, 0.7, 0.4]), \
        "Expected different scores or order"


def test_export_to_dataframe(dummy_data, tmp_path):
    create_test_classifier_csv_file(tmp_path)
    test_table: ResultsTable = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_analog_search_results.pickle'))[0]
    test_table.sqlite_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0], "tests/test_files/general_test_files/100_test_spectra.sqlite")
    test_table.query_spectrum.set("spectrum_nr", 1)
    test_table.classifier_csv_file_name = os.path.join(tmp_path, "test_csv_file")
    returned_dataframe = test_table.export_to_dataframe(5)
    assert isinstance(returned_dataframe, pd.DataFrame)
    assert list(returned_dataframe.columns) == column_names_for_output(True, True)
    # Check if one of the classifiers is filled in
    assert returned_dataframe["npc_pathway_results"][0] == "Amino acids and Peptides"
    assert len(returned_dataframe.index) == 5
    # Test if first row is correct
    print(returned_dataframe)
    np.testing.assert_array_almost_equal(
        list(returned_dataframe.iloc[0, :5]),
        [1, 0.56453, 33.25000, 907.0, 940.250],
        5)
    assert np.all(list(returned_dataframe.iloc[0, 5:12]) ==
                  ['KNGPFNUOXXLKCN', 'CCMSLIB00000001548', 'Hoiamide B',
                   'CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O',
                   'Organic compounds', 'Organic acids and derivatives', 'Peptidomimetics'])


def test_export_to_dataframe_with_additional_columns(tmp_path):
    create_test_classifier_csv_file(tmp_path)
    test_table: ResultsTable = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_analog_search_results.pickle'))[0]
    test_table.sqlite_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0], "tests/test_files/general_test_files/100_test_spectra.sqlite")
    test_table.classifier_csv_file_name = os.path.join(tmp_path, "test_csv_file")
    test_table.query_spectrum.set("spectrum_nr", 1)
    returned_dataframe = test_table.export_to_dataframe(5,
                                                        additional_metadata_columns=["charge"],
                                                        additional_ms2query_score_columns=["s2v_score", "ms2ds_score"])
    assert isinstance(returned_dataframe, pd.DataFrame)
    assert list(returned_dataframe.columns) == column_names_for_output(True, True, ["charge"],
                                                                       ["s2v_score", "ms2ds_score"])
    # Check if one of the classifiers is filled in
    assert returned_dataframe["npc_pathway_results"][0] == "Amino acids and Peptides"
    assert len(returned_dataframe.index) == 5
    # Test if first row is correct
    np.testing.assert_array_almost_equal(
        list(returned_dataframe.iloc[0, [0, 1, 2, 3, 4, 8, 9, 10]]),
        [1, 0.56453, 33.25000, 907.0, 940.250, 1, 0.99965, 0.92317],
        5)
    assert np.all(list(returned_dataframe.iloc[0, [5, 6, 7, 11, 12, 13, 14]]) ==
                       ['KNGPFNUOXXLKCN', 'CCMSLIB00000001548', 'Hoiamide B', 'CCC[C@@H](C)[C@@H]([C@H](C)[C@@H]1[C@H]([C@H](Cc2nc(cs2)C3=N[C@](CS3)(C4=N[C@](CS4)(C(=O)N[C@H]([C@H]([C@H](C(=O)O[C@H](C(=O)N[C@H](C(=O)O1)[C@@H](C)O)[C@@H](C)CC)C)O)[C@@H](C)CC)C)C)OC)C)O', 'Organic compounds', 'Organic acids and derivatives', 'Peptidomimetics'])