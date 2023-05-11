import os
import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from ms2query import ResultsTable
from ms2query.query_from_sqlite_database import SqliteLibrary
from ms2query.utils import column_names_for_output, load_pickled_file
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

    sqlite_test_file = "test_files/general_test_files/100_test_spectra.sqlite"

    return ms2deepscores, query_spectrum, SqliteLibrary(sqlite_test_file)


def test_table_init(dummy_data):
    ms2deepscores, query_spectrum, sqlite_library = dummy_data
    preselection_cut_off = 3
    table = ResultsTable(preselection_cut_off, ms2deepscores.iloc[:, 0], query_spectrum, sqlite_library)
    assert table.data.shape == (0, 8), \
        "Should have an empty data attribute"
    assert table.precursor_mz == 205.0, \
        "Expected different precursor m/z"


def test_table_preselect_ms2deepscore(dummy_data):
    ms2deepscores, query_spectrum, sqlite_library = dummy_data
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


def test_export_to_dataframe(dummy_data):
    test_table: ResultsTable = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_analog_search_results.pickle'))[0]
    # Add sqlite library as a patch to fix the test
    test_table.sqlite_library = dummy_data[2]
    test_table.query_spectrum.set("spectrum_nr", 1)
    returned_dataframe = test_table.export_to_dataframe(5)
    assert isinstance(returned_dataframe, pd.DataFrame)
    check_correct_results_csv_file(returned_dataframe.iloc[[0], :],
                                   column_names_for_output(True, True),
                                   nr_of_rows_to_check=1)


def test_export_to_dataframe_with_additional_columns(dummy_data):
    test_table: ResultsTable = load_pickled_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/expected_analog_search_results.pickle'))[0]
    test_table.sqlite_library = dummy_data[2]
    test_table.query_spectrum.set("spectrum_nr", 1)
    returned_dataframe = test_table.export_to_dataframe(5,
                                                        additional_metadata_columns=("charge",),
                                                        additional_ms2query_score_columns=("s2v_score", "ms2ds_score",))
    assert isinstance(returned_dataframe, pd.DataFrame)
    check_correct_results_csv_file(returned_dataframe.iloc[[0], :],
                                   column_names_for_output(True, True, ("charge",),
                                                           ("s2v_score", "ms2ds_score",)),
                                   nr_of_rows_to_check=1)
