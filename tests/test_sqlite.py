"""Tests all sqlite related functions

These functions are creating a new sqlite file with spectra data and
tanimoto scores (create_sqlite_database.py) and functions to retrieve
information from the sqlite database.
"""

import os
import pandas as pd
import numpy as np
import sqlite3
from matchms import Spectrum
from ms2query.app_helpers import load_pickled_file
from ms2query.create_sqlite_database import make_sqlfile_wrapper
from ms2query.query_from_sqlite_database import \
    get_tanimoto_score_for_inchikeys, get_spectra_from_sqlite


def test_making_sqlite_file(tmp_path):
    """Makes a temporary sqlite file and tests if it contains the correct info
    """
    # tmp_path is a fixture that makes sure a temporary file is created
    new_sqlite_file_name = os.path.join(tmp_path,
                                        "test_spectra_database.sqlite")

    path_to_test_files_sqlite_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files')

    reference_sqlite_file = os.path.join(path_to_test_files_sqlite_dir,
                                         "test_spectra_database.sqlite")
    # Create sqlite file, with 3 tables
    make_sqlfile_wrapper(new_sqlite_file_name,
                         os.path.join(path_to_test_files_sqlite_dir,
                                      "test_tanimoto_scores.npy"),
                         os.path.join(path_to_test_files_sqlite_dir,
                                      "first_10_spectra.pickle"),
                         os.path.join(path_to_test_files_sqlite_dir,
                                      "test_metadata_for_inchikey_order.csv"),
                         columns_dict={"parent_mass": "REAL"})

    # Test if file is made
    assert os.path.isfile(new_sqlite_file_name), \
        "Expected a file to be created"

    # Test if the file has the correct information
    get_table_names = \
        "SELECT name FROM sqlite_master WHERE type='table' order by name"
    conn1 = sqlite3.connect(new_sqlite_file_name)
    cur1 = conn1.cursor()
    table_names1 = cur1.execute(get_table_names).fetchall()

    conn2 = sqlite3.connect(reference_sqlite_file)
    cur2 = conn2.cursor()
    table_names2 = cur2.execute(get_table_names).fetchall()

    assert table_names1 == table_names2, \
        "Different sqlite tables are created than expected"

    for table_nr, table_name1 in enumerate(table_names1):
        table_name1 = table_name1[0]
        # Get column names and settings like primary key etc.
        table_info1 = cur1.execute(
            f"PRAGMA table_info({table_name1});").fetchall()
        table_info2 = cur2.execute(
            f"PRAGMA table_info({table_name1});").fetchall()
        assert table_info1 == table_info2, \
            f"Different column names or table settings " \
            f"were expected in table {table_name1}"
        column_names = [column_info[1] for column_info in table_info1]
        for column in column_names:
            # Get all rows from both tables
            rows_1 = cur1.execute(f"SELECT {column} FROM " +
                                  table_name1).fetchall()
            rows_2 = cur2.execute(f"SELECT {column} FROM " +
                                  table_name1).fetchall()
            error_msg = f"Different data was expected in column {column} " \
                f"in table {table_name1}. The wrong value is:"
            if column == "parent_mass":
                np.testing.assert_almost_equal(rows_1,
                                               rows_2,
                                               err_msg=error_msg,
                                               verbose=True)
            else:
                assert rows_1 == rows_2, error_msg
    conn1.close()
    conn2.close()


def test_get_tanimoto_scores():
    """Tests if the correct tanimoto scores are retrieved from sqlite file
    """
    path_to_test_files_sqlite_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files')
    sqlite_file_name = os.path.join(path_to_test_files_sqlite_dir,
                                    "test_spectra_database.sqlite")

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


def test_get_spectrum_data():
    """Tests if the correct spectrum data is returned from a sqlite file
    """
    path_to_test_files_sqlite_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files')
    sqlite_file_name = os.path.join(path_to_test_files_sqlite_dir,
                                    "test_spectra_database.sqlite")

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
        "Expected different spectrum to be loaded"
    assert original_spectra[2].__eq__(spectra_list[1]), \
        "Expected different spectrum to be loaded"
