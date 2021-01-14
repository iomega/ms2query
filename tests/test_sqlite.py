from ms2query.sqlite import create_table_structure, \
    add_spectra_to_database, get_spectra_from_sqlite
from ms2query.app_helpers import gather_test_json
import os
import json


def test_sqlite_functions(tmp_path):
    """Tests the functions create_table_structure, add_spectra_to_database
    and get_spectra_from sqlite"""

    column_type_dict = {"spectrum_id": "VARCHAR",
                        "source_file": "VARCHAR",
                        "ms_level": "INTEGER",
                        "key_that_does_not_exist": "VARCHAR"
                        }
    sqlite_file_name = os.path.join(tmp_path, "test_spectra_database.sqlite")
    # sqlite_file_name = "test_spectra_database.sqlite"
    # Give path to testfile
    path_to_json_test_file = gather_test_json("testspectrum_library.json")[0][
        "testspectrum_library.json"]

    create_table_structure(sqlite_file_name, column_type_dict)
    add_spectra_to_database(sqlite_file_name, path_to_json_test_file)
    spectra_list = get_spectra_from_sqlite(sqlite_file_name,
                                           ['CCMSLIB00000223876',
                                            'CCMSLIB00003138082'])

    # Test if the dictionaries of the spectra gotten from the sqlite file are
    # identical to one of the spectra in the json file, to make sure nothing is
    # lost in the process.
    spectra_json = json.load(open(path_to_json_test_file))
    for spectrum in spectra_list:
        assert spectrum in spectra_json, \
            "A spectrum is loaded that is not in the test json file"

    # Test if file is made
    assert os.path.isfile(sqlite_file_name), "Expected a file to be created"

    # Test if the output is of the right type
    assert isinstance(spectra_list, list), "Expected output to be list"
    assert isinstance(spectra_list[0], dict), \
        "Expected output to be list of dicts"
    assert isinstance(spectra_list[0]['spectrum_id'], str),\
        "Expected list of dicts, with a string for the key 'spectrum_id'"

