"""
Functions to create and retrieve information from sqlite files

Author: Niek de Jonge
"""

import sqlite3
import json
from typing import Dict, List
import ast

def create_table_structure(sqlite_file_name: str,
                           columns_dict: Dict[str, str],
                           table_name: str = "spectra"):
    """Creates a new sqlite file, with columns defined in columns_dict

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        table spectra is overwritten.
    columns_dict:
        Dictionary with as keys the column names and as values the sql datatype
    table_name:
        Name of the table that is created in the sqlite file,
        default = "spectra"
    """
    create_table_command = f"""
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
    """
    # add all columns with the type specified in columns_dict
    for column_header in columns_dict:
        create_table_command += column_header + " " \
                                + columns_dict[column_header] + ",\n"
    create_table_command += "full_json VARCHAR,\n"
    create_table_command += "PRIMARY KEY (spectrum_id));"

    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()
    cur.executescript(create_table_command)
    conn.commit()
    conn.close()


def add_spectra_to_database(sqlite_file_name: str,
                            json_spectrum_file_name: str,
                            table_name: str = "spectra"):
    """Creates a sqlite file containing the information from a json file.

     Args:
    -------
    sqlite_file_name:
        Name of sqlite file to which the spectra should be added.
    json_spectrum_file_name:
        File name of the json file that stores the spectra.
    table_name:
        Name of the table in the database to which the spectra should be added,
        default = "spectra"
    """
    spectra = json.load(open(json_spectrum_file_name))
    conn = sqlite3.connect(sqlite_file_name)

    # Get the column names in the sqlite file
    cur = conn.execute(f'select * from {table_name}')
    column_names = list(map(lambda x: x[0], cur.description))

    # Add the information of each spectrum to the sqlite file
    for spectrum in spectra:
        columns = ""
        values = ""
        # Check if there is a key for the spectrum that is the same as the
        # specified columns and if this is the case add this value to this
        # column.
        for column in column_names:
            if column in spectrum:
                # Add comma when it is not the first column that is added
                if len(columns) > 0:
                    columns += ", "
                    values += ", "
                columns += column
                values += '"' + spectrum[column] + '"'
        # Add the complete spectrum in json format to the column "full_json"
        if "full_json" in column_names:
            columns += ", full_json"
            values += f', "{spectrum}" '
        add_spectrum_command = f"INSERT INTO {table_name} ({columns}) values ({values})"

        cur = conn.cursor()
        cur.execute(add_spectrum_command)
        conn.commit()
    conn.close()

def get_spectra_from_sqlite(sqlite_file_name: str,
                            spectrum_id_list: List[str],
                            table_name: str = "spectra") -> List[dict]:
    """Returns a list with all metadata of spectrum_ids in spectrum_id_list

    Args:
    -------
    sqlite_file_name:
        File name of the sqlite file that contains the spectrum information
    spectrum_id_list:
        List of spectrum_id's of which the metadata should be returned
    table_name:
        Name of the table in the sqlite file that stores the spectrum data

    Returns:
    -------
    sqlite_spectra:
    """
    conn = sqlite3.connect(sqlite_file_name)


    sqlite_command = f"""SELECT full_json FROM {table_name} 
                    WHERE spectrum_id IN ('{"', '".join(map(str, spectrum_id_list))}')"""

    cur = conn.cursor()
    cur.execute(sqlite_command)

    sqlite_spectra = []
    for json_spectrum in cur:
        # Remove the "()" around the spectrum
        json_spectrum = json_spectrum[0]
        # Convert string to dictionary
        json_spectrum = ast.literal_eval(json_spectrum)
        sqlite_spectra.append(json_spectrum)
    conn.close()

    return sqlite_spectra

if __name__ == "__main__":
    # column_type_dict = {"spectrum_id": "VARCHAR",
    #                     "source_file": "VARCHAR",
    #                     "ms_level": "INTEGER",
    #                     }
    sqlite_file_name = "test_spectra_database.sqlite"
    # create_table_structure("test_spectra_database.sqlite",
    #                        column_type_dict)
    # add_spectra_to_database("test_spectra_database.sqlite",
    #                         "../tests/testspectrum_library.json")
    spectrum_list = get_spectra_from_sqlite(sqlite_file_name, ['CCMSLIB00000223876', 'CCMSLIB00003138082'])
    # print(spectrum_list)
    for spectrum in spectrum_list:
        print(spectrum)
        print(spectrum['spectrum_id'])
