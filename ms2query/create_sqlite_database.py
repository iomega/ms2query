"""
Functions to create sqlite file that contains 3 tables

The 3 tables are spectrum_data, inchikeys and tanimoto_scores.
Spectrum_data contains the peaks, intensities and metadata from all spectra.
Tanimoto_scores contains all tanimoto scores.
Inchikeys contains the order of inchikey14s, that can be used to find
corresponding indexes in the tanimoto_scores table.
"""

import gc
import io
import os
import sqlite3
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from matchms import Spectrum
from tqdm import tqdm
from ms2query.utils import load_pickled_file


def make_sqlfile_wrapper(sqlite_file_name: str,
                         tanimoto_scores_pickled_dataframe_file: str,
                         list_of_spectra: List[Spectrum],
                         columns_dict: Dict[str, str] = None,
                         progress_bars: bool = True,
                         spectrum_id_column_name: str = "spectrumid"):
    """Wrapper to create sqlite file with three tables.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        tables are added. If the tables in this sqlite file already exist, they
        will be overwritten.
    tanimoto_scores_pickled_dataframe_file:
        A pickled file with tanimoto scores. The column names and indexes are
        inchikey14s.
    list_of_spectra:
        A list with spectrum objects
    columns_dict:
        Dictionary with as keys columns that need to be added in addition to
        the default columns and as values the datatype. The defaults columns
        are spectrum_id, peaks, intensities and metadata. The additional
        columns should be the same names that are in the metadata dictionary,
        since these values will be automatically added in the function
        add_list_of_spectra_to_sqlite.
        Default = None results in the default columns.
    progress_bars:
        If progress_bars is True progress bars will be shown for the different
        parts of the progress.
    spectrum_id_column_name:
        The spectrum id column name is the name given to the column storing the
        spectrum ids. This is important since this name will be used to look
        up the spectrum id in the metadata. Per version of the data this
        differs between 'spectrum_id' and 'spectrumid'
    """
    # pylint: disable=too-many-arguments
    add_tanimoto_scores_to_sqlite(sqlite_file_name,
                                  tanimoto_scores_pickled_dataframe_file,
                                  list_of_spectra,
                                  progress_bars=progress_bars)
    create_table_structure(sqlite_file_name,
                           additional_columns_dict=columns_dict,
                           spectrum_column_name=spectrum_id_column_name)
    add_list_of_spectra_to_sqlite(sqlite_file_name,
                                  list_of_spectra,
                                  progress_bar=progress_bars)


def add_tanimoto_scores_to_sqlite(sqlite_file_name: str,
                                  tanimoto_scores_pickled_dataframe_file: str,
                                  list_of_spectra: List[Spectrum],
                                  temporary_tanimoto_file_name: str
                                  = "temporary_tanimoto_scores",
                                  progress_bars: bool = True):
    """Adds tanimoto scores and inchikey14s to sqlite table

    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        tables are added. If the tables in this sqlite file already exist, they
        will be overwritten.
    tanimoto_scores_pickled_dataframe_file:
        A pickled file with tanimoto scores. The column names and indexes are
        inchikey14s.
    list_of_spectra:
        List of spectrum objects
    temporary_tanimoto_file_name:
        The file name of a temporary .npy file that is created to memory
        efficiently read out the tanimoto scores. The file is deleted after
        finishing.
    progress_bars:
        If True progress bars will show the progress of the different steps
        in the process.
    """
    # todo instead of creating a real file and than deleting make a temporary
    #  file, not yet implemented since numpy memmap is not able to access the
    #  file in this case.

    temporary_tanimoto_file_name = os.path.join(os.getcwd(),
                                                temporary_tanimoto_file_name)
    assert not os.path.exists(temporary_tanimoto_file_name + ".npy"), \
        "A file already exists with the temporary file name you want to create"

    tanimoto_df = load_pickled_file(tanimoto_scores_pickled_dataframe_file)

    # Get spectra belonging to each inchikey14
    spectra_belonging_to_inchikey14 = \
        get_spectra_belonging_to_inchikey14(list_of_spectra)

    inchikeys_with_spectra = list(spectra_belonging_to_inchikey14.keys())
    inchikeys_with_spectra.sort()
    # Remove all inchikeys that do not have any matching spectra
    filtered_tanimoto_df = tanimoto_df.loc[inchikeys_with_spectra][
        inchikeys_with_spectra]

    inchikeys_order = filtered_tanimoto_df.index

    # Get closest related inchikey14s for each inchikey14
    closest_related_inchikey14s = \
        get_closest_related_inchikey14s(filtered_tanimoto_df, inchikeys_order)
    # Creates a sqlite table containing all tanimoto scores
    initialize_tanimoto_score_table(sqlite_file_name)

    assert not filtered_tanimoto_df.isnull().values.any(), \
        "No NaN values were expected in tanimoto scores"
    if progress_bars:
        print("Saving tanimoto scores to temporary .npy file")
    np.save(temporary_tanimoto_file_name, filtered_tanimoto_df.to_numpy())

    add_tanimoto_scores_to_sqlite_table(sqlite_file_name,
                                        temporary_tanimoto_file_name + ".npy",
                                        progress_bar=progress_bars)
    os.remove(temporary_tanimoto_file_name + ".npy")
    # Creates a table containing the identifiers belonging to each inchikey14
    # These identifiers correspond to the identifiers in tanimoto_scores
    create_inchikey_sqlite_table(sqlite_file_name,
                                 inchikeys_order,
                                 spectra_belonging_to_inchikey14,
                                 closest_related_inchikey14s,
                                 progress_bar=progress_bars)


def create_table_structure(sqlite_file_name: str,
                           additional_columns_dict: Dict[str, str] = None,
                           table_name: str = "spectrum_data",
                           spectrum_column_name: str = "spectrumid"):
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
    spectrum_column_name:
        The spectrum column name is the name given to the column storing the
        spectrum ids. This is important since this name will be used to look
        up the spectrum id in the metadata. Per version of the data this
        differs between 'spectrum_id' and 'spectrumid'
    """
    # Create a new datatype array for sqlite
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Initialize default columns
    if additional_columns_dict is None:
        additional_columns_dict = {}

    # Create default combined_columns_dict
    default_columns_dict = {spectrum_column_name: "TEXT",
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
    create_table_command += f"PRIMARY KEY ({spectrum_column_name}));"

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
                                  table_name: str = "spectrum_data",
                                  progress_bar: bool = True):
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
    progress_bar:
        If True a progress bar will show the progress.
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
    for spectrum in tqdm(list_of_spectra,
                         desc="Adding spectra to sqlite table",
                         disable=not progress_bar):
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


def create_inchikey_sqlite_table(
        file_name: str,
        ordered_inchikey_list: List[str],
        spectra_belonging_to_inchikey: Dict[str, List[str]],
        closest_related_inchikeys: Dict[str, List[Tuple[str, float]]],
        progress_bar: bool = True):
    """Creates a table storing the identifiers belonging to the inchikey14s

    Overwrites the table if it already exists. The column specified in
    col_name_inchikey becomes the primary key. The datatype of
    col_name_inchikey is set to TEXT. The created table is used to look up
    identifiers used in the table created by the function
    initialize_tanimoto_score_table.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite file to which the table should be added.
    ordered_inchikey14_list:
        List with the first 14 symbols of inchikeys, the inchikey14s will be
        stored in sqlite in the same order.
    spectra_belonging_to_inchikey:
        A dictionary storing the spectra belonging to each inchikey
    closest_related_inchikeys:
        A dictionary storing the closest related inchikeys and the tanimoto
        score between the two inchikeys.
    table_name:
        Name of the table in the database that should store the (order of) the
        inchikey14s. Default = inchikeys.
    col_name_inchikey:
        Name of the column in the table containing the inchikey14s.
        Default = inchikey
    progress_bar:
        If True a progress bar is shown.
    """
    table_name = 'inchikeys'
    col_name_inchikey = 'inchikey'

    conn = sqlite3.connect(file_name)

    # Initialize table
    create_table_command = f""";
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
        {col_name_inchikey} TEXT,
        spectrum_ids_belonging_to_inchikey14 TEXT,
        closest_related_inchikeys TEXT,
        PRIMARY KEY ({col_name_inchikey})
        );
    """
    cur = conn.cursor()
    cur.executescript(create_table_command)
    conn.commit()

    # Fill table
    for inchikey14 in tqdm(ordered_inchikey_list,
                           desc="Adding inchikey14s to sqlite table",
                           disable=not progress_bar):
        matching_spectrum_ids = str(spectra_belonging_to_inchikey[inchikey14])
        add_row_to_table_command = \
            f"""INSERT INTO {table_name} 
            values ("{inchikey14}", 
            "{matching_spectrum_ids}",
            "{str(closest_related_inchikeys[inchikey14])}");"""
        cur.execute(add_row_to_table_command)
    conn.commit()
    conn.close()


def get_spectra_belonging_to_inchikey14(spectra: List[Spectrum]
                                        ) -> Dict[str, List[str]]:
    """Returns a dictionary with the spectrum_ids belonging to each inchikey14

    Args:
    -----
    inchikey14s:
        List of inchikey14s
    spectra:
        List of spectrum objects
    """
    spectra_belonging_to_inchikey14 = {}
    for spectrum in spectra:
        inchikey14_of_spectrum = spectrum.get("inchikey")[:14]
        spectrum_id = spectrum.get("spectrumid")
        if inchikey14_of_spectrum in spectra_belonging_to_inchikey14:
            spectra_belonging_to_inchikey14[inchikey14_of_spectrum].append(spectrum_id)
        else:
            spectra_belonging_to_inchikey14[inchikey14_of_spectrum] = [spectrum_id]
    return spectra_belonging_to_inchikey14


def get_closest_related_inchikey14s(tanimoto_scores: pd.DataFrame,
                                    ordered_inchikeys):
    """Returns the closest related inchikeys based on tanimoto scores"""
    closest_related_inchikey14s_dict = {}
    for inchikey in ordered_inchikeys:
        closest_related_inchikey14s_dict[inchikey] = []

        closest_related_inchikey14s = tanimoto_scores[inchikey].nlargest(10)
        for closest_related_inchikey14 in \
                closest_related_inchikey14s.iteritems():
            closest_related_inchikey14s_dict[inchikey].append(
                closest_related_inchikey14)
    return closest_related_inchikey14s_dict


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
        that are the identifier of inchikey14s. Default = 'identifier_1'
    col_name_identifier2:
        Name of the second column of the table, this column will store numbers
        that are the identifier of inchikey14s. Default = 'identifier_2'
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
        that are the identifier of inchikey14s. Default = 'identifier_1'
    col_name_identifier2:
        Name of the second column of the table, this column will store numbers
        that are the identifier of inchikey14s. Default = 'identifier_2'
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
    # inchikey14s
    list_of_numbers = []
    for i in range(len(tanimoto_score_matrix[0])):
        list_of_numbers.append(i+1)

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
        df[col_name_identifier1] = np.array(len(row) * [row_nr+1])
        df[col_name_identifier2] = np.array(list_of_numbers[:len(row)])

        df.rename(columns={0: col_name_score}, inplace=True)
        # Add dataframe to table in sqlite file
        convert_dataframe_to_sqlite(df, sqlite_file_name)
    # del and gc.collect are needed to later delete the file
    del tanimoto_score_matrix
    gc.collect()


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
