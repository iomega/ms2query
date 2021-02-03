"""
Functions to create sqlite file that contains 3 tables

The 3 tables are spectrum_data, inchikeys and tanimoto_scores.
Spectrum_data contains the peaks, intensities and metadata from all spectra.
Tanimoto_scores contains all tanimoto scores.
Inchikeys contains the order of inchikeys, that can be used to find
corresponding indexes in the tanimoto_scores table.
"""

import io
from tqdm import tqdm
from typing import Dict, List
import pandas as pd
import numpy as np
import sqlite3
from matchms import Spectrum
from ms2query.app_helpers import load_pickled_file


def make_sqlfile_wrapper(sqlite_file_name: str,
                         npy_file_location: str,
                         pickled_file_name: str,
                         csv_file_with_inchikey_order: str,
                         columns_dict: Dict[str, str] = None
                         ):
    """Wrapper to create sqlite file with three tables.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        tables are added. If the tables in this sqlite file already exist, they
        will be overwritten.
    npy_file_location:
        The file location of an .npy file. This file is expected to contain the
        tanimoto scores and have an equal amount of rows and columns equal to
        the number of inchikeys in csv_file_with_inchikey_order.
    pickled_file_name:
        This file is expected to contain a list of matchms.Spectrum.
    csv_file_with_inchikey_order:
        This csv file is expected to contain inchikeys on the second column
        starting from the second row. These inchikeys should be ordered in the
        same way the tanimoto scores are ordered in the npy_file_location.
    columns_dict:
        Dictionary with as keys columns that need to be added in addition to
        the default columns and as values the datatype. The defaults columns
        are spectrum_id, peaks, intensities and metadata. The additional
        columns should be the same names that are in the metadata dictionary,
        since these values will be automatically added in the function
        add_list_of_spectra_to_sqlite.
        Default = None results in the default columns.
    """

    # Creates a sqlite table containing all tanimoto scores
    initialize_tanimoto_score_table(sqlite_file_name)
    add_tanimoto_scores_to_sqlite_table(sqlite_file_name, npy_file_location)

    # Creates a table containing the identifiers belonging to each inchikey
    # These identifiers correspond to the identifiers in tanimoto_scores
    create_inchikey_sqlite_table(sqlite_file_name,
                                 csv_file_with_inchikey_order)

    # Creates a sqlite table with the metadata, peaks and intensities
    list_of_spectra = load_pickled_file(pickled_file_name)
    create_table_structure(sqlite_file_name,
                           additional_columns_dict=columns_dict)
    add_list_of_spectra_to_sqlite(sqlite_file_name, list_of_spectra)


def create_table_structure(sqlite_file_name: str,
                           additional_columns_dict: Dict[str, str] = None,
                           table_name: str = "spectrum_data"):
    """Creates a new sqlite file, with columns defined in combined_columns_dict

    On default the columns spectrum_id, peaks, intensities and metadata are
    created. The column spectrum_id will be the primary key. Extra columns can
    be added by specifying these

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists a new
        table is added. If the table with the name specified by table_name
        already exists, then this table is overwritten.
    additional_columns_dict:
        Dictionary with as keys columns that need to be added in addition to
        the default columns and as values the datatype. The defaults columns
        are spectrum_id, peaks, intensities and metadata. The additional
        columns should be the same names that are in the metadata dictionary,
        since these values will be automatically added in the function
        add_list_of_spectra_to_sqlite.
        Default = None results in the default columns.
    table_name:
        Name of the table that is created in the sqlite file,
        default = "spectrum_data"
    """
    # Create a new datatype array for sqlite
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Initialize default columns
    if additional_columns_dict is None:
        additional_columns_dict = {}

    # Create default combined_columns_dict
    default_columns_dict = {"spectrum_id": "TEXT",
                            "peaks": "array",
                            "intensities": "array",
                            "metadata": "TEXT"}

    combined_columns_dict = {**default_columns_dict, **additional_columns_dict}
    create_table_command = f"""
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
    """
    # add all columns with the type specified in combined_columns_dict
    for column_header in combined_columns_dict:
        create_table_command += column_header + " " \
                                + combined_columns_dict[column_header] + ",\n"
    create_table_command += "PRIMARY KEY (spectrum_id));"

    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()
    cur.executescript(create_table_command)
    conn.commit()
    conn.close()


def adapt_array(arr: np.array) -> sqlite3.Binary:
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
    """Adds the data of a list of spectra to a sqlite file

    The spectrum_id is the primary key and the data is stored in the columns
    peaks, itensities and metadata. If additional columns are specified in
    the table, the values stored for this column name in metadata are stored
    in the table.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file to which the spectrum data is added. The sqlite
        file is expected to already contain a table with the name specified by
        table_name.
    list_of_spectra:
        The Spectra of which the data is added to the sqlite file.
    table_name:
        Name of the table to which the spectrum data is added. The table is
        expected to already contain the default columns spectrum_id, peaks,
        intensities and metadata.
        default = "spectrum_data"
    """
    conn = sqlite3.connect(sqlite_file_name)

    # Get the column names in the sqlite file
    cur = conn.execute(f'select * from {table_name}')
    column_names = list(map(lambda x: x[0], cur.description))

    # Create a string with placeholders for the add spectrum command
    value_placeholders = ""
    for column in column_names:
        value_placeholders += ':' + column + ", "

    add_spectrum_command = f"""INSERT INTO {table_name}
                           values ({value_placeholders[:-2]})"""

    # Add the data of each spectrum to the sqlite table
    for spectrum in list_of_spectra:
        peaks = spectrum.peaks.mz
        intensities = spectrum.peaks.intensities
        metadata = spectrum.metadata

        # Give the value for the default columns
        value_dict = {'peaks': peaks,
                      "intensities": intensities,
                      "metadata": str(metadata)}
        # Gets the data for addition columns from metadata
        for column in column_names:
            if column not in ['peaks', 'intensities', 'metadata']:
                if column in metadata:
                    value_dict[column] = str(metadata[column])
                else:
                    value_dict[column] = ""

        cur = conn.cursor()
        cur.execute(add_spectrum_command, value_dict)

    conn.commit()
    conn.close()


def create_inchikey_sqlite_table(file_name: str,
                                 csv_file_with_inchikey_order: str,
                                 table_name: str = 'inchikeys',
                                 col_name_inchikey: str = 'inchikey'):
    """Creates a table storing the identifiers belonging to the inchikeys

    Overwrites the table if it already exists. The column specified in
    col_name_inchikey becomes the primary key. The datatype of
    col_name_inchikey is set to TEXT. The created table is used to look up
    identifiers used in the table created by the function
    initialize_tanimoto_score_table.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite file to which the table should be added.
    csv_file_with_inchikey_order:
        This csv file is expected to contain inchikeys on the second column
        starting from the second row. These inchikeys should be ordered in the
        same way the tanimoto scores are ordered in the tanimoto_scores table.
    table_name:
        Name of the table in the database that should store the (order of) the
        inchikeys. Default = inchikeys.
    col_name_inchikey:
        Name of the column in the table containing the inchikeys.
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

    # Get the ordered inchikeys from the csv file
    ordered_inchikey_list = get_inchikey_order(csv_file_with_inchikey_order)

    # Fill table
    for inchikey in ordered_inchikey_list:
        add_row_to_table_command = f"""INSERT INTO {table_name} 
                                    values ('{inchikey}');"""
        cur.execute(add_row_to_table_command)
    conn.commit()
    conn.close()


def get_inchikey_order(metadata_csv_file: str) -> List[str]:
    """Return list of Inchi14s in same order as in metadata_file

    Args:
    ------
    metadata_csv_file:
        path to metadata file, expected format is csv, with inchikeys in the
        second column, starting from the second row.
    """
    with open(metadata_csv_file, 'r') as inf:
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
                                        col_name_score: str = 'tanimoto_score',
                                        progress_bar: bool = True,
                                        ):
    """Adds tanimoto scores from npy file to sqlite table

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
    progress_bar:
        Show progress bar if set to True. Default is True.
    """
    # pylint: disable=too-many-arguments
    tanimoto_score_matrix = np.load(
        npy_file_path,
        mmap_mode='r')

    # Create a list of consecutive numbers, later used as identifiers for the
    # inchikeys
    list_of_numbers = []
    for i in range(len(tanimoto_score_matrix[0])):
        list_of_numbers.append(i)

    # Create a dataframe and add to sqlite table row by row
    for row_nr, row in enumerate(
            tqdm(tanimoto_score_matrix,
                 desc="Adding tanimoto scores to sqlite file",
                 disable=not progress_bar)):

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
