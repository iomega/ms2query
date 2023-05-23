import os

import pandas as pd

from ms2query.clean_and_filter_spectra import normalize_and_filter_peaks_multiple_spectra
from ms2query.create_new_library.calculate_tanimoto_scores import calculate_tanimoto_scores_unique_inchikey, \
    calculate_highest_tanimoto_score
from ms2query.utils import load_matchms_spectrum_objects_from_file, load_pickled_file


def test_calculate_tanimoto_scores_unique_inchikey(path_to_general_test_files, hundred_test_spectra):
    tanimoto_df = calculate_tanimoto_scores_unique_inchikey(hundred_test_spectra, hundred_test_spectra)
    expected_tanimoto_df = load_pickled_file(os.path.join(path_to_general_test_files,
                                                          "100_test_spectra_tanimoto_scores.pickle"))
    assert isinstance(tanimoto_df, pd.DataFrame), "Expected a pandas dataframe"
    pd.testing.assert_frame_equal(tanimoto_df, expected_tanimoto_df, check_exact=False, atol=1e-5)


def test_calculate_tanimoto_scores_unique_inchikey_not_symmetric(path_to_general_test_files, hundred_test_spectra):
    spectra_2 = hundred_test_spectra[:10]
    tanimoto_df = calculate_tanimoto_scores_unique_inchikey(hundred_test_spectra, spectra_2)

    unique_inchikey_2 = set([spectrum.get("inchikey")[:14] for spectrum in spectra_2])
    expected_tanimoto_df = load_pickled_file(os.path.join(path_to_general_test_files,
                                                          "100_test_spectra_tanimoto_scores.pickle")
                                             ).loc[:, sorted(unique_inchikey_2)]
    assert isinstance(tanimoto_df, pd.DataFrame), "Expected a pandas dataframe"
    pd.testing.assert_frame_equal(tanimoto_df, expected_tanimoto_df,
                                  check_exact=False, atol=1e-5)


def test_calculate_closest_related_inchikeys(tmp_path, path_to_general_test_files, hundred_test_spectra):
    list_of_spectra = normalize_and_filter_peaks_multiple_spectra(hundred_test_spectra)
    result = calculate_highest_tanimoto_score(list_of_spectra, list_of_spectra, 10)
    assert isinstance(result, dict), "expected a dictionary"
    assert len(result) == 61
    for inchikey in result:
        assert isinstance(inchikey, str), "Expected inchikey to be string"
        assert len(inchikey) == 14, "Expected an inchikey of length 14"
        assert isinstance(result[inchikey], list), "Expected a dictionary with as keys a list"
        assert len(result[inchikey]) == 10
        for score in result[inchikey]:
            assert isinstance(score, tuple)
            assert isinstance(score[0], str), "Expected inchikey to be string"
            assert len(score[0]) == 14, "Expected an inchikey of length 14"
            assert isinstance(score[1], float)


def test_calculate_highest_tanimoto_score(path_to_general_test_files, hundred_test_spectra):
    list_of_spectra = normalize_and_filter_peaks_multiple_spectra(hundred_test_spectra)
    results = calculate_highest_tanimoto_score(list_of_spectra[:2], list_of_spectra, 2)
    assert results == {'IYDKWWDUBYWQGF': [('IYDKWWDUBYWQGF', 1.0), ('XCGGFKXEWPNQFS', 0.5548738922972052)],
                       'KNGPFNUOXXLKCN': [('KNGPFNUOXXLKCN', 1.0), ('NZZSDJHUISSTSC', 0.7344213649851632)]}
