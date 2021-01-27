"""
Functions to create sqlite file that contains 3 tables

The 3 tables are spectrum_data, inchikeys and tanimoto_scores.
Spectrum_data contains the peaks, intensities and metadata from all spectra.
Tanimoto_scores contains all tanimoto scores.
Inchikeys contains the order of inchikeys, that can be used to find
corresponding indexes in the tanimoto_scores table.
"""

import sqlite3
from typing import Dict, List
import pandas as pd
import numpy as np
import time
from ms2query.app_helpers import load_pickled_file
from matchms.Spectrum import Spectrum
import io


def make_sqlfile_wrapper(sqlite_file_name: str,
                         npy_file_location: str,
                         pickled_file_name: str,
                         csv_file_with_inchikey_order:str
                         ):
    """Wrapper to create sqlite file with three tables.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        tables are added. If the tables in this sqlite file already exist, they
        will be overwritten.
    json_spectrum_file_name:
        File location of the json file that stores the spectra.
    columns_dict:
        Dictionary with as keys columns that need to be added in addition to
        the default columns (peaks, intensities, metadata and spectrum_id) and
        as values the datatype. The column names should correspond to keys in
        the metadata.

    """

    # Creates a sqlite table containing all tanimoto scores
    initialize_tanimoto_score_table(sqlite_file_name)
    add_tanimoto_scores_to_sqlite_table(sqlite_file_name, npy_file_location)

    # Creates a table containing the identifiers belonging to each inchikey
    create_inchikey_sqlite_table(sqlite_file_name, csv_file_with_inchikey_order)

    # Creates a sqlite table with the metadata, peaks and intensities
    list_of_spectra = load_pickled_file(pickled_file_name)
    create_table_structure(sqlite_file_name)
    add_list_of_spectra_to_sqlite(sqlite_file_name, list_of_spectra)


def create_table_structure(sqlite_file_name: str,
                           columns_dict: Dict[str, str] = None,
                           table_name: str = "spectrum_data"):
    """Creates a new sqlite file, with columns defined in columns_dict

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        table spectra is overwritten.
    columns_dict:
        Dictionary with as keys columns that need to be added and as values the
        datatype. Default = None results in the columns spectrum_id, peaks,
        intensities and metadata. These column names always have to be in
        columns_dict.
    table_name:
        Name of the table that is created in the sqlite file,
        default = "spectrum_data"
    """
    # Create a new datatype array for sqlite
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)


    if columns_dict is None:
        columns_dict = {"spectrum_id": "TEXT",
                        "peaks": "array",
                        "intensities": "array",
                        "metadata": "TEXT"}

    create_table_command = f"""
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
    """
    # add all columns with the type specified in columns_dict
    for column_header in columns_dict:
        create_table_command += column_header + " " \
                                + columns_dict[column_header] + ",\n"
    create_table_command += "PRIMARY KEY (spectrum_id));"

    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()
    cur.executescript(create_table_command)
    conn.commit()
    conn.close()


def adapt_array(arr):
    """Converts array to binary format, so it can be stored in sqlite

    By running this command:
    sqlite3.register_adapter(np.ndarray, adapt_array)
    This function will be called everytime a np.ndarray is loaded into a sqlite
    file.

    Found at: http://stackoverflow.com/a/31312102/190597 (SoulNibbler)

    Args:
    -------
    arr:
        A np ndarray
        """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def add_list_of_spectra_to_sqlite(sqlite_file_name: str,
                                  list_of_spectra: List[Spectrum],
                                  table_name: str = "spectrum_data"):
    conn = sqlite3.connect(sqlite_file_name)
    # Get the column names in the sqlite file
    cur = conn.execute(f'select * from {table_name}')
    column_names = list(map(lambda x: x[0], cur.description))

    value_placeholders = ""
    for column in column_names:
        value_placeholders += ':' + column + ", "

    add_spectrum_command = f"""INSERT INTO {table_name}
                           values ({value_placeholders[:-2]})"""

    for nr, spectrum in enumerate(list_of_spectra):
        peaks = spectrum.peaks.mz
        intensities = spectrum.peaks.intensities
        metadata = spectrum.metadata

        value_dict = {'peaks': peaks,
                      "intensities": intensities,
                      "metadata": str(metadata)}
        for column in column_names:
            if column not in ['peaks', 'intensities', 'metadata']:
                if column in metadata:
                    value_dict[column] = str(metadata[column])
                else:
                    value_dict[column] = ""

        cur = conn.cursor()
        cur.execute(add_spectrum_command, value_dict)
        if nr % 100 == 0:
            print(nr)
    conn.commit()
    conn.close()


def create_inchikey_sqlite_table(file_name: str,
                                 csv_file_with_inchikey_order: str,
                                 table_name: str = 'inchikeys',
                                 col_name_inchikey: str = 'inchikey'):
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
    ordered_inchikey_list = get_inchikey_order(csv_file_with_inchikey_order)

    for inchikey in ordered_inchikey_list:
        add_row_to_table_command = f"""INSERT INTO {table_name} 
                                    values ('{inchikey}');"""
        cur.execute(add_row_to_table_command)
    conn.commit()
    conn.close()


def get_inchikey_order(metadata_file: str) -> List[str]:
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
    # make_sqlfile_wrapper("../downloads/data_all_inchikeys.sqlite",
    #                      "../downloads/similarities_AllInchikeys14_daylight2048_jaccard.npy",
    #                      "../downloads/gnps_positive_ionmode_cleaned_by_matchms_and_lookups.pickle")
    # list_of_spectra = load_pickled_file("../downloads/gnps_positive_ionmode_cleaned_by_matchms_and_lookups.pickle")
    # list_of_spectra = list_of_spectra[:10]
    # outfile = open("../tests/test_files_sqlite/first_10_spectra.pickle", "wb")
    # pickle.dump(list_of_spectra, outfile)
    # outfile.close()
    # list_of_spectra = load_pickled_file("../tests/test_files_sqlite/first_10_spectra.pickle")
    # print(list_of_spectra)
    pass

