"""
Functions to create and retrieve information from sqlite files
"""

import sqlite3
import json
from typing import Dict, List
import ast
from matchms.importing.load_from_json import dict2spectrum
from matchms.Spectrum import Spectrum


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
        File location of the json file that stores the spectra.
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

            # Add the complete spectrum in json format to the column full_json
            elif column == "full_json":
                if len(columns) > 0:
                    columns += ", "
                    values += ", "
                columns += "full_json"
                values += f'"{spectrum}"'
            else:
                print(f"No value found for column: {column} in "
                      f"spectrum {spectrum['spectrum_id']}")
        add_spectrum_command = f"INSERT INTO {table_name} " \
                               + f"({columns}) values ({values})"

        cur = conn.cursor()
        cur.execute(add_spectrum_command)
        conn.commit()
    conn.close()


def get_spectra_from_sqlite(sqlite_file_name: str,
                            spectrum_id_list: List[str],
                            table_name: str = "spectra") -> List[Spectrum]:
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
    list_of_spectra_dict:
    """
    conn = sqlite3.connect(sqlite_file_name)

    sqlite_command = f"""SELECT full_json FROM {table_name} 
                    WHERE spectrum_id 
                    IN ('{"', '".join(map(str, spectrum_id_list))}')"""

    cur = conn.cursor()
    cur.execute(sqlite_command)

    list_of_spectra_dict = []
    for json_spectrum in cur:
        # Remove the "()" around the spectrum
        json_spectrum = json_spectrum[0]
        # Convert string to dictionary
        json_spectrum = ast.literal_eval(json_spectrum)
        list_of_spectra_dict.append(json_spectrum)
    conn.close()

    # Convert to matchms.Spectrum.Spectrum object
    spectra_list = []
    for spectrum in list_of_spectra_dict:
        spectra_list.append(dict2spectrum(spectrum))
    return spectra_list
