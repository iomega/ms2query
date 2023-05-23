"""Tests all sqlite related functions

These functions are creating a new sqlite file with spectra data and
tanimoto scores (create_sqlite_database.py) and functions to retrieve
information from the sqlite database.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from ms2query.create_new_library.create_sqlite_database import make_sqlfile_wrapper
from ms2query.clean_and_filter_spectra import normalize_and_filter_peaks_multiple_spectra
from ms2query.utils import load_pickled_file, column_names_for_output
from ms2query.create_new_library.add_classifire_classifications import convert_to_dataframe


def check_sqlite_files_are_equal(new_sqlite_file_name, reference_sqlite_file, check_metadata=True):
    """Raises an error if the two sqlite files are not equal"""
    # Test if file is made
    assert os.path.isfile(new_sqlite_file_name), \
        "Expected a file to be created"
    assert os.path.isfile(reference_sqlite_file), \
        "The reference file given does not exist"

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
                f"in table {table_name1}. \n Expected {rows_2} \n got {rows_1}"
            if column == "precursor_mz":
                np.testing.assert_almost_equal(rows_1,
                                               rows_2,
                                               err_msg=error_msg,
                                               verbose=True)
            elif column == "metadata" and not check_metadata:
                pass
            else:
                assert len(rows_1) == len(rows_2)
                for i in range(len(rows_1)):
                    assert rows_1[i] == rows_2[i], f"Different data was expected in column {column} row {i}" \
                                                   f"in table {table_name1}. \n Expected {rows_2[i]} \n got {rows_1[i]}"
    conn1.close()
    conn2.close()


def test_making_sqlite_file_without_classes(tmp_path, hundred_test_spectra, path_to_general_test_files):
    """Makes a temporary sqlite file and tests if it contains the correct info
    """
    # tmp_path is a fixture that makes sure a temporary file is created
    new_sqlite_file_name = os.path.join(tmp_path,
                                        "test_spectra_database.sqlite")

    reference_sqlite_file = os.path.join(path_to_general_test_files,
                                         "..",
                                         "backwards_compatibility",
                                         "100_test_spectra_without_classes.sqlite")

    list_of_spectra = normalize_and_filter_peaks_multiple_spectra(hundred_test_spectra)

    # Create sqlite file, with 3 tables
    make_sqlfile_wrapper(new_sqlite_file_name,
                         list_of_spectra,
                         columns_dict={"precursor_mz": "REAL"})
    check_sqlite_files_are_equal(new_sqlite_file_name, reference_sqlite_file, check_metadata=False)


def test_making_sqlite_file_with_compound_classes(tmp_path, path_to_general_test_files, hundred_test_spectra):
    """Makes a temporary sqlite file and tests if it contains the correct info
    """
    def generate_compound_classes(spectra):
        inchikeys = {spectrum.get("inchikey")[:14] for spectrum in spectra}
        inchikey_results_list = []
        for inchikey in inchikeys:
            inchikey_results_list.append([inchikey, "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        compound_class_df = convert_to_dataframe(inchikey_results_list)
        return compound_class_df
    # tmp_path is a fixture that makes sure a temporary file is created
    new_sqlite_file_name = os.path.join(tmp_path,
                                        "test_spectra_database.sqlite")

    reference_sqlite_file = os.path.join(path_to_general_test_files,
                                         "100_test_spectra.sqlite")

    list_of_spectra = normalize_and_filter_peaks_multiple_spectra(hundred_test_spectra)

    # Create sqlite file, with 3 tables
    make_sqlfile_wrapper(new_sqlite_file_name,
                         list_of_spectra,
                         columns_dict={"precursor_mz": "REAL"},
                         compound_classes=generate_compound_classes(spectra=list_of_spectra))

    check_sqlite_files_are_equal(new_sqlite_file_name, reference_sqlite_file, check_metadata=False)


def test_get_metadata_from_sqlite(sqlite_library):
    spectra_id_list = [0, 1]

    result = sqlite_library.get_metadata_from_sqlite(
        spectra_id_list,
        spectrum_id_storage_name="spectrumid")
    assert isinstance(result, dict), "Expected dictionary as output"
    assert len(result) == len(spectra_id_list), \
        "Expected the same number of results as the spectra_id_list"
    for spectrum_id in spectra_id_list:
        assert spectrum_id in result.keys(), \
            f"The spectrum_id {spectrum_id} was expected as key"
        metadata = result[spectrum_id]
        assert isinstance(metadata, dict), \
            "Expected metadata to be stored as dict"
        for key in metadata.keys():
            assert isinstance(key, str), \
                "Expected keys of metadata to be string"
            assert isinstance(metadata[key], (str, float, int, list, tuple)), \
                f"Expected values of metadata to be string {metadata[key]}"


def test_get_ionization_mode_library(sqlite_library):
    ionization_mode = sqlite_library.get_ionization_mode_library()
    assert ionization_mode == "positive"


def test_get_classes_inchikeys(sqlite_library):
    test_inchikeys = ["IYDKWWDUBYWQGF", "KNGPFNUOXXLKCN"]
    classes = sqlite_library.get_classes_inchikeys(test_inchikeys)
    expected_classes = pd.DataFrame([["IYDKWWDUBYWQGF", "b", "c", "d", "e", "f", "g", "h", "i"],
                                     ["KNGPFNUOXXLKCN", "b", "c", "d", "e", "f", "g", "h", "i"]],
                                    columns=["inchikey"] + column_names_for_output(return_non_classifier_columns=False,
                                                                                   return_classifier_columns=True))
    pd.testing.assert_frame_equal(expected_classes, classes)


def test_contains_class_annotiation(sqlite_library):
    assert sqlite_library.contains_class_annotation(), "contains_class_annotation is expected to return True"
