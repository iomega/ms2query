import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from ms2query import ResultsTable
from ms2query.utils import column_names_for_output
from tests.test_utils import check_correct_results_csv_file


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

    return ms2deepscores, query_spectrum


@pytest.fixture
def create_results_tables(ms2library, test_spectra):
    result_tables = ms2library.analog_search_return_results_tables(test_spectra,
                                                                   preselection_cut_off=20,
                                                                   store_ms2deepscore_scores=True)
    assert isinstance(result_tables, list)
    for result_table in result_tables:
        assert isinstance(result_table, ResultsTable)
    return result_tables


def test_table_init(dummy_data, sqlite_library):
    ms2deepscores, query_spectrum = dummy_data
    preselection_cut_off = 3
    table = ResultsTable(preselection_cut_off, ms2deepscores.iloc[:, 0], query_spectrum, sqlite_library)
    assert table.data.shape == (0, 8), \
        "Should have an empty data attribute"
    assert table.precursor_mz == 205.0, \
        "Expected different precursor m/z"


def test_table_preselect_ms2deepscore(dummy_data, sqlite_library):
    ms2deepscores, query_spectrum = dummy_data
    preselection_cut_off = 3
    table = ResultsTable(preselection_cut_off, ms2deepscores.iloc[:, 0], query_spectrum, sqlite_library)
    table.preselect_on_ms2deepscore()
    assert table.data.shape == (3, 8), "Should have different data table"
    assert np.all(table.data.spectrum_ids.values ==
                  ['XXXXXXXXXXXXXC', 'XXXXXXXXXXXXXB', 'XXXXXXXXXXXXXD']), \
        "Expected different spectrum IDs or order"
    assert np.all(table.data.ms2ds_score.values ==
                  [0.99, 0.7, 0.4]), \
        "Expected different scores or order"


def test_export_to_dataframe(dummy_data, sqlite_library, create_results_tables):
    test_table: ResultsTable = create_results_tables[0]
    # Add sqlite library as a patch to fix the test
    test_table.sqlite_library = sqlite_library
    test_table.query_spectrum.set("spectrum_nr", 1)
    returned_dataframe = test_table.export_to_dataframe(5)
    assert isinstance(returned_dataframe, pd.DataFrame)
    check_correct_results_csv_file(returned_dataframe.iloc[[0], :],
                                   column_names_for_output(True, True),
                                   nr_of_rows_to_check=1)


def test_export_to_dataframe_with_additional_columns(dummy_data, sqlite_library, create_results_tables):
    test_table = create_results_tables[0]
    test_table.sqlite_library = sqlite_library
    test_table.query_spectrum.set("spectrum_nr", 1)
    returned_dataframe = test_table.export_to_dataframe(5,
                                                        additional_metadata_columns=("charge",),
                                                        additional_ms2query_score_columns=("s2v_score", "ms2ds_score",))
    assert isinstance(returned_dataframe, pd.DataFrame)
    check_correct_results_csv_file(returned_dataframe.iloc[[0], :],
                                   column_names_for_output(True, True, ("charge",),
                                                           ("s2v_score", "ms2ds_score",)),
                                   nr_of_rows_to_check=1)
