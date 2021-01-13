"""
Test file to implement sqlite database for Spectrum data

Author: Niek de Jonge
"""

import sqlite3
import json
from typing import Dict


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

        if "full_json" in column_names:
            columns += ", full_json"
            values += f', "{spectrum}" '
        add_spectrum_command = f"INSERT INTO {table_name} ({columns}) values ({values})"

        cur = conn.cursor()
        cur.execute(add_spectrum_command)
        conn.commit()
    conn.close()


if __name__ == "__main__":
    column_type_dict = {"spectrum_id": "VARCHAR",
                        "source_file": "VARCHAR",
                        "ms_level": "INTEGER",
                        }
    create_table_structure("test_spectra_database.sqlite",
                           column_type_dict)
    add_spectra_to_database("test_spectra_database.sqlite",
                            "../tests/testspectrum_library.json")
