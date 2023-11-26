import os
import sqlite3
import numpy as np
import pandas as pd
import pytest
from gensim.models import Word2Vec
from ms2deepscore.models import load_model as load_ms2ds_model
from ms2query.clean_and_filter_spectra import (
    normalize_and_filter_peaks, normalize_and_filter_peaks_multiple_spectra)
from ms2query.create_new_library.add_classifire_classifications import \
    _convert_to_dataframe
from ms2query.create_new_library.library_files_creator import (
    LibraryFilesCreator, create_ms2ds_embeddings, create_s2v_embeddings)


def test_give_already_used_file_name(tmp_path, path_to_general_test_files, hundred_test_spectra):
    already_existing_file = os.path.join(tmp_path, "ms2query_library.sqlite")
    with open(already_existing_file, "w") as file:
        file.write("test")

    with pytest.raises(FileExistsError):
        LibraryFilesCreator(hundred_test_spectra, tmp_path)


def test_create_ms2ds_embeddings(tmp_path, path_to_general_test_files,
                                 hundred_test_spectra,
                                 expected_ms2ds_embeddings,
                                 ms2deepscore_model_file_name):
    """Tests store_ms2ds_embeddings"""
    library_spectra = [normalize_and_filter_peaks(s) for s in hundred_test_spectra if s is not None]
    embeddings = create_ms2ds_embeddings(ms2ds_model=load_ms2ds_model(ms2deepscore_model_file_name),
                                         list_of_spectra=library_spectra)
    pd.testing.assert_frame_equal(embeddings, expected_ms2ds_embeddings,
                                  check_exact=False,
                                  atol=1e-5)


def test_create_s2v_embeddings(tmp_path, path_to_general_test_files, hundred_test_spectra,
                              expected_s2v_embeddings,
                               spec2vec_model_file_name):
    """Tests store_ms2ds_embeddings"""
    library_spectra = [normalize_and_filter_peaks(s) for s in hundred_test_spectra if s is not None]

    embeddings = create_s2v_embeddings(Word2Vec.load(spec2vec_model_file_name), library_spectra)
    pd.testing.assert_frame_equal(embeddings, expected_s2v_embeddings,
                                  check_exact=False,
                                  atol=1e-5)

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


def test_create_sqlite_file_with_embeddings(tmp_path,
                                                  hundred_test_spectra,
                                                  ms2deepscore_model_file_name,
                                                  spec2vec_model_file_name,
                                                  sqlite_library):
    """Makes a temporary sqlite file and tests if it contains the correct info
    """
    def generate_compound_classes(spectra):
        inchikeys = {spectrum.get("inchikey")[:14] for spectrum in spectra}
        inchikey_results_list = []
        for inchikey in inchikeys:
            inchikey_results_list.append([inchikey, "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        compound_class_df = _convert_to_dataframe(inchikey_results_list)
        return compound_class_df
    new_sqlite_file_name = os.path.join(tmp_path,
                                        "test_spectra_database.sqlite")

    list_of_spectra = normalize_and_filter_peaks_multiple_spectra(hundred_test_spectra)
    library_creator = \
        LibraryFilesCreator(library_spectra=list_of_spectra,
                            sqlite_file_name=new_sqlite_file_name,
                            s2v_model_file_name=spec2vec_model_file_name,
                            ms2ds_model_file_name=ms2deepscore_model_file_name,
                            compound_classes=generate_compound_classes(spectra=list_of_spectra))
    library_creator.create_sqlite_file()

    check_sqlite_files_are_equal(new_sqlite_file_name, sqlite_library.sqlite_file_name, check_metadata=False)
