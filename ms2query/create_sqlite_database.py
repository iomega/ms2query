"""
Functions to create and retrieve information from sqlite files
"""

import sqlite3
import json
from typing import Dict, List
import pandas as pd
import numpy as np
import time


# todo test metadata maker, for full file, is in pickled format. Change
#  function to take list of spectrums as input. Change test function with load
#  from json, so this is still functional.
def make_sqlfile_wrapper(sqlite_file_name: str,
                         columns_dict: Dict[str, str],
                         json_spectrum_file_name: str,
                         npy_file_location: str
                         ):
    """Wrapper to create sqlite file with three tables.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        tables are added. If the tables in this sqlite file already exist, they
        will be overwritten.
    columns_dict:
        Dictionary with as keys the column names and as values the sql datatype
    json_spectrum_file_name:
        File location of the json file that stores the spectra.

    """

    create_table_structure(sqlite_file_name, columns_dict)
    add_spectra_to_database(sqlite_file_name, json_spectrum_file_name)
    create_inchikey_sqlite_table(sqlite_file_name)
    initialize_tanimoto_score_table(sqlite_file_name)
    add_tanimoto_scores_to_sqlite_table(sqlite_file_name, npy_file_location)


def create_table_structure(sqlite_file_name: str,
                           columns_dict: Dict[str, str],
                           table_name: str = "metadata"):
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
        default = "metadata"
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
                            table_name: str = "metadata"):
    """Creates a sqlite file containing the information from a json file.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite file to which the spectra should be added.
    json_spectrum_file_name:
        File location of the json file that stores the spectra.
    table_name:
        Name of the table in the database to which the spectra should be added,
        default = "metadata"
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


def create_inchikey_sqlite_table(file_name: str,
                                 table_name: str = 'inchikeys',
                                 col_name_inchikey: str = 'inchikeys',
                                 col_name_identifier: str = 'identifier'):
    """Creates a table storing the identifiers belonging to the inchikeys

    Removes the table if it already exists and makes the col_name_inchikey
    the primary key. The datatype of col_name_inchikey is set
    to TEXT and the datatype of col_name_identifier is set to INTEGER. The
    created table is used to look up identifiers used in the table created by
    the function initialize_tanimoto_score_table.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite file to which the table should be added.
    table_name:
        Name of the table in the database that should store the tanimoto
        scores. Default = inchikeys.
    col_name_inchikey:
        Name of the first column in the table containing the inchikeys.
        Default = inchikeys
    col_name_identifier:
        Name of the second column in the table containing the identifiers.
        Default = identifier.

    """

    conn = sqlite3.connect(file_name)

    # Initialize table
    create_table_command = f""";
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
        {col_name_inchikey} TEXT,
        PRIMARY KEY ({col_name_inchikey})
        );
    """
    cur = conn.cursor()
    cur.executescript(create_table_command)
    conn.commit()

    # Fill table
    ordered_inchikey_list = get_inchikey_order()

    for inchikey in ordered_inchikey_list:
        add_row_to_table_command = f"""INSERT INTO {table_name} 
                                    values ('{inchikey}');"""
        cur.execute(add_row_to_table_command)
    conn.commit()
    conn.close()


def get_inchikey_order(metadata_file: str =
                       "../downloads/metadata_AllInchikeys14.csv"
                       ) -> List[str]:
    """Return list of Inchi14s in same order as in metadata_file

    Args:
    ------
    metadata_file:
        path to metadata file, expected format is csv, with inchikeys in the
        second column, starting from the second row. Default =
        "../downloads/metadata_AllInchikeys14.csv"
    """
    with open(metadata_file, 'r') as inf:
        inf.readline()
        inchi_list = []
        for line in inf:
            line = line.strip().split(',')
            inchi_list.append(line[1])
    return inchi_list


def initialize_tanimoto_score_table(sqlite_file_name: str,
                                    table_name: str = 'tanimoto_scores',
                                    col_name_identifier1: str = 'identifier_1',
                                    col_name_identifier2: str = 'identifier_2',
                                    col_name_score: str = 'tanimoto_score'):
    """Initializes a table for the tanimoto scores.

    Removes the table if it already exists and makes the col_name_identifiers
    together a composite key. The datatype of both col_name_identifiers is set
    to INTEGER and datatype of col_name_score is set to REAL.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite file to which the table should be added.
    table_name:
        Name of the table in the database that should store the tanimoto
        scores. Default = tanimoto_scores.
    col_name_identifier1:
        Name of the first column of the table, this column will store numbers
        that are the identifier of inchikeys. Default = 'identifier_1'
    col_name_identifier2:
        Name of the second column of the table, this column will store numbers
        that are the identifier of inchikeys. Default = 'identifier_2'
    col_name_score:
        Name of the third column of the table, this column will store the
        tanimoto scores. Default = 'tanimoto_score'
    """
    create_table_command = f""";
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
        {col_name_identifier1} INTEGER,
        {col_name_identifier2} INTEGER,
        {col_name_score} REAL,
        PRIMARY KEY ({col_name_identifier1}, {col_name_identifier2})
        );
    """
    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()
    cur.executescript(create_table_command)
    conn.commit()
    conn.close()


def add_tanimoto_scores_to_sqlite_table(sqlite_file_name: str,
                                        npy_file_path: str,
                                        col_name_identifier1: str =
                                        'identifier_1',
                                        col_name_identifier2: str =
                                        'identifier_2',
                                        col_name_score: str = 'tanimoto_score'
                                        ):
    """Adds tanimoto scores to sqlite table

    The tanimoto scores are added to a sqlite file with in the first and
    second column the identifiers that correspond to the index of the rows and
    columns in the .npy file. The third row contains the tanimoto score. All
    duplicates are removed, before adding to the sqlite table.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite file to which the table should be added.
    npy_file_path:
        Path to the numpy file containing the tanimoto scores. The numpy file
        is expected to contain a matrix filled with floats with nr of columns =
        nr of rows.
    col_name_identifier1:
        Name of the first column of the table, this column will store numbers
        that are the identifier of inchikeys. Default = 'identifier_1'
    col_name_identifier2:
        Name of the second column of the table, this column will store numbers
        that are the identifier of inchikeys. Default = 'identifier_2'
    col_name_score:
        Name of the third column of the table, this column will store the
        tanimoto scores. Default = 'tanimoto_score'
    """
    tanimoto_score_matrix = np.load(
        npy_file_path,
        mmap_mode='r')

    # Create a list of consecutive numbers, later used as identifiers for the
    # inchikeys
    list_of_numbers = []
    for i in range(len(tanimoto_score_matrix[0])):
        list_of_numbers.append(i)

    start_time = time.time()
    # Create a dataframe and add to sqlite table row by row
    for row_nr, row in enumerate(tanimoto_score_matrix):
        # Remove duplicates
        row = row[:row_nr+1]
        # Transform to pd.Dataframe
        df = pd.DataFrame(row)
        # Add columns with identifiers (they correspond to the indexes of the
        # rows and columns in the npy file
        df[col_name_identifier1] = np.array(len(row) * [row_nr])
        df[col_name_identifier2] = np.array(list_of_numbers[:len(row)])

        df.rename(columns={0: col_name_score}, inplace=True)
        # Add dataframe to table in sqlite file
        convert_dataframe_to_sqlite(df, sqlite_file_name)

        if row_nr % 100 == 0:
            print(row_nr)
    print("--- %s seconds ---" % (time.time() - start_time))


def convert_dataframe_to_sqlite(spectrum_dataframe: pd.DataFrame,
                                file_name: str,
                                table_name: str = 'tanimoto_scores'):
    """Adds the content of a dataframe to an sqlite table.

    Args:
    -------
    spectrum_dataframe:
        The dataframe that has to be added to the sqlite table.
    file_name:
        Name of sqlite file to which the table should be added.
    table_name:
        Name of the table in the database that should store the tanimoto
        scores. Default = tanimoto_scores."""

    connection = sqlite3.connect(file_name)
    spectrum_dataframe.to_sql(table_name, connection,
                              method='multi',
                              chunksize=30000,
                              if_exists="append",
                              index=False)
    connection.commit()
    connection.close()


if __name__ == "__main__":
    column_type_dict = {"spectrum_id": "VARCHAR",
                        "source_file": "VARCHAR",
                        "ms_level": "INTEGER",
                        }

    make_sqlfile_wrapper("test_file_3_tables.sqlite",
                         column_type_dict,
                         "../tests/testspectrum_library.json",
                         "../downloads/similarities_AllInchikeys14_daylight2048_jaccard.npy")
