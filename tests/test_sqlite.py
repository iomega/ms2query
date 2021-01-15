from ms2query.sqlite import create_table_structure, \
    add_spectra_to_database, get_spectra_from_sqlite
from ms2query.app_helpers import gather_test_json
import os
import json
from matchms.Spectrum import Spectrum


def test_sqlite_functions(tmp_path):
    """Tests the functions create_table_structure, add_spectra_to_database
    and get_spectra_from sqlite"""

    column_type_dict = {"spectrum_id": "VARCHAR",
                        "source_file": "VARCHAR",
                        "ms_level": "INTEGER",
                        "key_that_does_not_exist": "VARCHAR"
                        }
    sqlite_file_name = os.path.join(tmp_path, "test_spectra_database.sqlite")
    spectra_id_list = ['CCMSLIB00000223876', 'CCMSLIB00003138082']

    # Get path to json testfile
    path_to_json_test_file = gather_test_json("testspectrum_library.json")[0][
        "testspectrum_library.json"]

    create_table_structure(sqlite_file_name, column_type_dict)
    add_spectra_to_database(sqlite_file_name, path_to_json_test_file)
    spectra_list = get_spectra_from_sqlite(sqlite_file_name, spectra_id_list)

    # Test if file is made
    assert os.path.isfile(sqlite_file_name), "Expected a file to be created"

    # Test if the output is of the right type
    assert isinstance(spectra_list, list), "Expected a list"
    assert isinstance(spectra_list[0], Spectrum), \
        "Expected a list with matchms.Spectrum.Spectrum objects"

    # Test if the right number of spectra are returned
    assert len(spectra_list) == 2, "Expected only 2 spectra"

    # Test if the correct metadata is returned for both spectra
    spectra_json = json.load(open(path_to_json_test_file))
    for key in spectra_list[0].metadata:
        assert spectra_json[0][key] == spectra_list[0].metadata[key], \
            "Expected different metadata for this spectrum"
    for key in spectra_list[1].metadata:
        assert spectra_json[2][key] == spectra_list[1].metadata[key], \
            "Expected different metadata for this spectrum"


