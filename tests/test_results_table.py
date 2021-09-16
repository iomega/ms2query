import numpy as np
import pytest
import pandas as pd
from matchms import Spectrum
from ms2query import ResultsTable


@pytest.fixture
def dummy_data():
    ms2deepscores = pd.DataFrame(np.array([0.2, 0.7, 0.99, 0.4]),
                                 index = ["XXXXXXXXXXXXXA",
                                          "XXXXXXXXXXXXXB",
                                          "XXXXXXXXXXXXXC",
                                          "XXXXXXXXXXXXXD"])

    query_spectrum = Spectrum(mz = np.array([100.0]),
                              intensities = np.array([1.0]),
                              metadata = {"parent_mass": 205.0})

    return ms2deepscores, query_spectrum


def test_table_init(dummy_data):
    ms2deepscores, query_spectrum = dummy_data
    preselection_cut_off = 3
    table = ResultsTable(preselection_cut_off,
                         ms2deepscores.iloc[:, 0],
                         query_spectrum)
    assert table.data.shape == (0, 11), \
        "Should have an empty data attribute"
    assert table.parent_mass == 205.0, \
        "Expected different parent mass"


def test_table_preselect_ms2deepscore(dummy_data):
    ms2deepscores, query_spectrum = dummy_data
    preselection_cut_off = 3
    table = ResultsTable(preselection_cut_off,
                         ms2deepscores.iloc[:, 0],
                         query_spectrum)
    table.preselect_on_ms2deepscore()
    assert table.data.shape == (3, 11), "Should have different data table"
    assert np.all(table.data.spectrum_ids.values == \
                  ['XXXXXXXXXXXXXC', 'XXXXXXXXXXXXXB', 'XXXXXXXXXXXXXD']), \
        "Expected different spectrum IDs or order"
    assert np.all(table.data.ms2ds_score.values == \
                  [0.99, 0.7, 0.4]), \
        "Expected different scores or order"


def test_add_parent_masses(dummy_data):
    ms2deepscores, query_spectrum = dummy_data
    preselection_cut_off = 2
    table = ResultsTable(preselection_cut_off,
                         ms2deepscores.iloc[:, 0],
                         query_spectrum)
    table.add_parent_masses(np.array([190.0, 199.2, 200.0, 201.0]), 0.8)
    expected = np.array([0.03518437, 0.27410813, 0.32768, 0.4096])

    assert table.data.shape == (4, 11), \
        "Should have different data table"
    assert np.all(np.isclose(table.data.mass_similarity.values,
                             expected, atol=1e-7)), \
        "Expected different scores or order"
