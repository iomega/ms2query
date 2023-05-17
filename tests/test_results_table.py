import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from ms2query import ResultsTable


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
