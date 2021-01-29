import os
import pandas as pd
import numpy as np
from matchms import Spectrum
from ms2query.app_helpers import load_pickled_file
from ms2query.create_sqlite_database import make_sqlfile_wrapper
from ms2query.query_from_sqlite_database import \
    get_tanimoto_score_for_inchikeys, get_spectra_from_sqlite


def test_sqlite_functions_wrapper(tmp_path):
    """Tests all sqlite related functions

    These functions are creating a new sqlite file with spectra data and
    tanimoto scores (create_sqlite_database.py) and functions to retrieve
    information from the sqlite database.
    """
    sqlite_file_name = os.path.join(tmp_path, "test_spectra_database.sqlite")

    path_to_test_files_sqlite_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files_sqlite')

    # Creates a temporary sqlite file with test data.
    make_sqlite_file(sqlite_file_name, path_to_test_files_sqlite_dir)
    # Test if the correct tanimoto score data is retrieved from the sqlite file
    get_tanimoto_scores(sqlite_file_name, path_to_test_files_sqlite_dir)
    # Test if the correct spectrum data is loaded from the sqlite file
    get_spectrum_data(sqlite_file_name, path_to_test_files_sqlite_dir)

def make_sqlite_file(sqlite_file_name, path_to_test_files_sqlite_dir):
    """Makes a sqlite file and tests if it was made

    Args:
    ------
    sqlite_file_name:
        The file name of the temporary test sqlite file
    path_to_test_files_sqlite_dir:
        The path from this directory to the directory with test files
    """
    # Create sqlite file, with 3 tables
    make_sqlfile_wrapper(sqlite_file_name,
                         os.path.join(path_to_test_files_sqlite_dir,
                                      "test_tanimoto_scores.npy"),
                         os.path.join(path_to_test_files_sqlite_dir,
                                      "first_10_spectra.pickle"),
                         os.path.join(path_to_test_files_sqlite_dir,
                                      "test_metadata_for_inchikey_order.csv"))

    # Test if file is made
    assert os.path.isfile(sqlite_file_name), "Expected a file to be created"

def get_tanimoto_scores(sqlite_file_name, path_to_test_files_sqlite_dir):
    """Tests if the correct tanimoto scores are retrieved from sqlite file

    Args:
    ------
    sqlite_file_name:
        The file name of the temporary test sqlite file
    path_to_test_files_sqlite_dir:
        The path from this directory to the directory with test files
    """
    test_inchikeys = ['MYHSVHWQEVDFQT',
                      'BKAWJIRCKVUVED',
                      'CXVGEDCSTKKODG']
    tanimoto_score_dataframe = get_tanimoto_score_for_inchikeys(
        test_inchikeys,
        sqlite_file_name)

    scores_in_test_file = np.load(os.path.join(path_to_test_files_sqlite_dir,
                                               "test_tanimoto_scores.npy"))

    expected_dataframe = pd.DataFrame(scores_in_test_file,
                                      index=test_inchikeys,
                                      columns=test_inchikeys)

    assert expected_dataframe.equals(tanimoto_score_dataframe), \
        "Expected different tanimoto scores, or columns/index names"

def get_spectrum_data(sqlite_file_name, path_to_test_files_sqlite_dir):
    """Tests if the correct spectrum data is returned from a sqlite file

    Args:
    ------
    sqlite_file_name:
        The file name of the temporary test sqlite file
    path_to_test_files_sqlite_dir:
        The path from this directory to the directory with test files
    """
    spectra_id_list = ['CCMSLIB00000001547', 'CCMSLIB00000001549']
    spectra_list = get_spectra_from_sqlite(sqlite_file_name, spectra_id_list)

    # Test if the output is of the right type
    assert isinstance(spectra_list, list), "Expected a list"
    assert isinstance(spectra_list[0], Spectrum), \
        "Expected a list with matchms.Spectrum.Spectrum objects"

    # Test if the right number of spectra are returned
    assert len(spectra_list) == 2, "Expected only 2 spectra"

    # Test if the correct spectra are loaded
    pickled_file_name = os.path.join(path_to_test_files_sqlite_dir,
                                     "first_10_spectra.pickle")
    original_spectra = load_pickled_file(pickled_file_name)
    assert original_spectra[0].__eq__(spectra_list[0]), \
        "Expected different spectrum"
    assert original_spectra[2].__eq__(spectra_list[1]), \
        "Expected different spectrum"
