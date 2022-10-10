import os
import pandas as pd
import pytest
from ms2query.create_new_library.train_ms2deepscore import calculate_tanimoto_scores
from ms2query.utils import load_matchms_spectrum_objects_from_file, load_pickled_file


@pytest.fixture
def path_to_general_test_files() -> str:
    return os.path.join(
        os.getcwd(),
        './test_files/general_test_files')


def test_calculate_tanimoto_scores(path_to_general_test_files):
    spectra = load_matchms_spectrum_objects_from_file(
        os.path.join(path_to_general_test_files, '100_test_spectra.pickle'))
    tanimoto_df = calculate_tanimoto_scores(spectra)
    expected_tanimoto_df = load_pickled_file(os.path.join(path_to_general_test_files,
                                                          "100_test_spectra_tanimoto_scores.pickle"))
    assert isinstance(tanimoto_df, pd.DataFrame), "Expected a pandas dataframe"
    pd.testing.assert_frame_equal(tanimoto_df, expected_tanimoto_df, check_exact=False, atol=1e-5)
