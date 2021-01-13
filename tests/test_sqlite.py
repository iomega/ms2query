from ms2query.sqlite import create_table_structure, \
    add_spectra_to_database, get_spectra_from_sqlite
from ms2query.app_helpers import gather_test_json


def test_sqlite_functions():
    """Tests the functions create_table_structure, add_spectra_to_database
    and get_spectra_from sqlite"""
    column_type_dict = {"spectrum_id": "VARCHAR",
                        "source_file": "VARCHAR",
                        "ms_level": "INTEGER",
                        "key_that_does_not_exist": "VARCHAR"
                        }
    sqlite_file_name = "test_spectra_database.sqlite"

    # Give path to testfile
    path_to_json_test_file = gather_test_json("testspectrum_library.json")[0][
        "testspectrum_library.json"]

    create_table_structure(sqlite_file_name, column_type_dict)
    add_spectra_to_database(sqlite_file_name, path_to_json_test_file)
    spectra_list = get_spectra_from_sqlite(sqlite_file_name,
                                           ['CCMSLIB00000223876',
                                            'CCMSLIB00003138082'])
    assert isinstance(spectra_list, list), "Expected output to be list"
    assert isinstance(spectra_list[0], dict), \
        "Expected output to be list of dicts"
    assert isinstance(spectra_list[0]['spectrum_id'], str),\
        "Expected list of dicts, with a string for the key 'spectrum_id'"
