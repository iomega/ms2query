from ms2query.sqlite import create_table_structure, \
    add_spectra_to_database, get_spectra_from_sqlite


def test_create_table_structure_and_add_spectra_to_database():
    """Tests the functions create_table_structure, add_spectra_to_database
    and get_spectra_from sqlite"""
    column_type_dict = {"spectrum_id": "VARCHAR",
                        "source_file": "VARCHAR",
                        "ms_level": "INTEGER",
                        "key_that_does_not_exist": "VARCHAR"
                        }
    sqlite_file_name = "test_spectra_database.sqlite"
    create_table_structure(sqlite_file_name, column_type_dict)
    add_spectra_to_database(sqlite_file_name,
                            "../tests/testspectrum_library.json")
    spectra_list = get_spectra_from_sqlite(sqlite_file_name,
                                           ['CCMSLIB00000223876',
                                            'CCMSLIB00003138082'])
    assert(isinstance(spectra_list, list))
    assert(isinstance(spectra_list[0], dict))
    assert(isinstance(spectra_list[0]['spectrum_id'], str))

