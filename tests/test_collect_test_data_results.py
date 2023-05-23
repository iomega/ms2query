import os
import pandas as pd
import pytest
import numpy as np
from matchms import Spectrum
from ms2query.benchmarking.collect_test_data_results import (generate_test_results_ms2query,
                                                             get_all_ms2ds_scores,
                                                             select_highest_ms2ds_in_mass_range,
                                                             get_modified_cosine_score_results,
                                                             get_cosines_score_results,
                                                             create_optimal_results,
                                                             create_random_results,
                                                             generate_test_results)

from ms2query.utils import load_matchms_spectrum_objects_from_file, load_json_file


@pytest.fixture
def local_test_spectra():
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


def test_generate_test_results_ms2query(ms2library, local_test_spectra):
    result = generate_test_results_ms2query(ms2library, local_test_spectra)
    np.testing.assert_almost_equal(result[0], (0.5645, 0.003861003861003861, False))
    np.testing.assert_almost_equal(result[1], (0.409, 0.010610079575596816, False))

    # test if a spectrum that does not pass the tests is not added to the results
    local_test_spectra[0] = local_test_spectra[0].set("precursor_mz", None)
    local_test_spectra[0] = local_test_spectra[0].set("pepmass", None)
    result = generate_test_results_ms2query(ms2library, local_test_spectra)
    assert result[0] is None
    np.testing.assert_almost_equal(result[1], (0.409, 0.010610079575596816, False))


def test_get_all_ms2ds_scores(ms2library, local_test_spectra):
    result = get_all_ms2ds_scores(ms2library.ms2ds_model,
                                  ms2library.ms2ds_embeddings,
                                  local_test_spectra)
    assert isinstance(result, pd.DataFrame)
    assert float(result.iloc[0, 0]).__round__(5) == 0.76655


def test_select_highest_ms2ds_in_mass_range(ms2library, local_test_spectra):
    ms2ds = get_all_ms2ds_scores(ms2library.ms2ds_model,
                                 ms2library.ms2ds_embeddings,
                                 local_test_spectra)

    # test with mass 100 preselection
    result = select_highest_ms2ds_in_mass_range(ms2ds,
                                                local_test_spectra,
                                                ms2library.sqlite_library,
                                                100)
    np.testing.assert_almost_equal(result[0], (0.8492529314990583, 0.003861003861003861, False))
    np.testing.assert_almost_equal(result[1], (0.6413115894635883, 0.013745704467353952, False))

    # test without mass preselection
    result_without_mass_range = select_highest_ms2ds_in_mass_range(ms2ds,
                                                                   local_test_spectra,
                                                                   ms2library.sqlite_library,
                                                                   None)
    np.testing.assert_almost_equal(result_without_mass_range[0], (0.8492529314990583, 0.003861003861003861, False))
    np.testing.assert_almost_equal(result_without_mass_range[1], (0.8514114889698237, 0.007292616226071103, False))

    # test with mass preselection resulting in 0 and 1 library spectra within mass range
    result = select_highest_ms2ds_in_mass_range(ms2ds,
                                                local_test_spectra,
                                                ms2library.sqlite_library,
                                                5.56)
    np.testing.assert_almost_equal(result[0], (0.7368508, 0.004461, False))
    assert result[1] is None


def test_get_modified_cosine_score_results(local_test_spectra, hundred_test_spectra):
    results = get_modified_cosine_score_results(hundred_test_spectra, local_test_spectra, 100)
    np.testing.assert_almost_equal(results,
                                   [(0.434789196140529, 0.003861003861003861, False),
                                    (0.4955472245596076, 0.007866273352999017, False)], decimal=4)
    # Test if no error happens when only 1 or 0 library spectra within mass range
    results = get_modified_cosine_score_results(hundred_test_spectra, local_test_spectra, 5.56)
    np.testing.assert_almost_equal(results[0],
                                   (0.0, 0.0044609665427509295, False), decimal=4)
    assert results[1] is None


def test_get_cosines_score_results(local_test_spectra, hundred_test_spectra):
    result = get_cosines_score_results(hundred_test_spectra, local_test_spectra, 100, 0.05, 3)
    np.testing.assert_almost_equal(result,
                                   [(0.434789196140529, 0.0058997050147492625, False),
                                    (0.4955472245596076, 0.007866273352999017, False)], decimal=4)
    # Test if no error happens when only 1 or 0 library spectra within mass range
    result = get_cosines_score_results(hundred_test_spectra, local_test_spectra, 5.56, 0.05, 0)
    np.testing.assert_almost_equal(result[0],
                                   (0.0, 0.004461, False), decimal=4)
    assert result[1] is None


def test_create_optimal_results(local_test_spectra):
    results = create_optimal_results(local_test_spectra, local_test_spectra)
    assert results == [(1.0, 1.0, True), (1.0, 1.0, True)]


def test_random_results(local_test_spectra):
    results = create_random_results(local_test_spectra,
                                    local_test_spectra)
    assert len(results) == len(local_test_spectra)


def test_generate_test_results(local_test_spectra,
                               ms2library,
                               tmp_path, hundred_test_spectra):
    generate_test_results(ms2library,
                          hundred_test_spectra,
                          local_test_spectra,
                          tmp_path)
    files_made = os.listdir(tmp_path)
    assert set(files_made) == {'cosine_score_100_da_test_results.json', 'modified_cosine_score_100_Da_test_results.json',
                          'ms2deepscore_test_results_100_Da.json', 'ms2deepscore_test_results_all.json',
                          'ms2query_test_results.json', 'optimal_results.json', 'random_results.json'}
    for file in files_made:
        result = load_json_file(os.path.join(tmp_path, file))
        assert isinstance(result, list)
