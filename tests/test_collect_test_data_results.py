import os

import pandas as pd
import pytest
import numpy as np
from matchms import Spectrum
from ms2query.ms2library import MS2Library
from ms2query.benchmarking.collect_test_data_results import (generate_test_results_ms2query,
                                                             get_all_ms2ds_scores,
                                                             select_highest_ms2ds_in_mass_range,
                                                             get_modified_cosine_score_results,
                                                             get_cosines_score_results,
                                                             create_optimal_results)
from tests.test_use_files_without_spectrum_id import ms2library_without_spectrum_id
from ms2query.utils import load_matchms_spectrum_objects_from_file


@pytest.fixture
def test_spectra():
    """Returns a list with two spectra

    The spectra are created by using peaks from the first two spectra in
    100_test_spectra.pickle, to make sure that the peaks occur in the s2v
    model. The other values are random.
    """
    spectrum1 = Spectrum(
        mz=np.array([808.27356, 872.289917, 890.246277, 891.272888, 894.326416, 904.195679], dtype="float"),
        intensities=np.array([0.11106008, 0.12347332, 0.16352988, 0.17101522, 0.17312992, 1.], dtype="float"),
        metadata={'pepmass': (907.0, None),
                  'spectrumid': 'CCMSLIB00000001760',
                  'precursor_mz': 907.0,
                  'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N',
                  'smiles': "CCCC",
                  'charge': 1})
    spectrum2 = Spectrum(
        mz=np.array([538.003174, 539.217773, 556.030396, 599.352783, 851.380859, 852.370605], dtype="float"),
        intensities=np.array([0.28046377, 0.28900242, 0.31933114, 0.32199162, 0.71323034, 1.], dtype="float"),
        metadata={'pepmass': (928.0, None),
                  'spectrumid': 'CCMSLIB00000001761',
                  'precursor_mz': 928.0,
                  'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N',
                  'smiles': "CCCCC"
                  })
    return [spectrum1, spectrum2]


def test_generate_test_results_ms2query(ms2library_without_spectrum_id, test_spectra, tmp_path):
    result = generate_test_results_ms2query(ms2library_without_spectrum_id,
                                            test_spectra,
                                            os.path.join(tmp_path, "temporary.csv"))
    assert result[0] == (0.5645, 0.003861003861003861, False)
    assert result[1] == (0.409, 0.010610079575596816, False)


def test_get_all_ms2ds_scores(ms2library_without_spectrum_id, test_spectra):
    result = get_all_ms2ds_scores(ms2library_without_spectrum_id.ms2ds_model,
                                  ms2library_without_spectrum_id.ms2ds_embeddings,
                                  test_spectra)
    assert isinstance(result, pd.DataFrame)
    assert float(result.iloc[0, 0]).__round__(5) == 0.76655


def test_select_highest_ms2ds_in_mass_range(ms2library_without_spectrum_id, test_spectra):
    ms2ds = get_all_ms2ds_scores(ms2library_without_spectrum_id.ms2ds_model,
                                 ms2library_without_spectrum_id.ms2ds_embeddings,
                                 test_spectra)
    result = select_highest_ms2ds_in_mass_range(ms2ds,
                                                test_spectra,
                                                ms2library_without_spectrum_id.sqlite_file_name,
                                                100)
    assert result[0] == (0.8492529314990583, 0.003861003861003861, False)
    assert result[1] == (0.6413115894635883, 0.013745704467353952, False)

    result_without_mass_range = select_highest_ms2ds_in_mass_range(ms2ds,
                                                                   test_spectra,
                                                                   ms2library_without_spectrum_id.sqlite_file_name,
                                                                   None)
    assert result_without_mass_range[0] == (0.8492529314990583, 0.003861003861003861, False)
    assert result_without_mass_range[1] == (0.8514114889698237, 0.007292616226071103, False)


def test_get_modified_cosine_score_results(test_spectra):
    library_spectra = load_matchms_spectrum_objects_from_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files/100_test_spectra.pickle'))
    results = get_modified_cosine_score_results(library_spectra, test_spectra, 100)
    assert results == [(0.434789196140529, 0.003861003861003861, False),
                       (0.4955472245596076, 0.007866273352999017, False)]


def test_get_cosines_score_results(test_spectra):
    library_spectra = load_matchms_spectrum_objects_from_file(os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files/100_test_spectra.pickle'))
    result = get_cosines_score_results(library_spectra, test_spectra, 100, 0.05, 3)
    assert result == [(0.434789196140529, 0.0058997050147492625, False),
                      (0.4955472245596076, 0.007866273352999017, False)]


def test_create_optimal_results(test_spectra):
    results = create_optimal_results(test_spectra, test_spectra)
    assert results == [(1.0, 1.0, True), (1.0, 1.0, True)]


if __name__ == "__main__":
    pass
