"""
This script is not needed for normally running MS2Query, it is only needed to generate a new library or to train
new models
"""

import sqlite3
from typing import Dict, List
from matchms import Spectrum
from tqdm import tqdm

from ms2query.create_new_library.calculate_tanimoto_scores import calculate_highest_tanimoto_score


def make_sqlfile_wrapper(sqlite_file_name: str,
                         list_of_spectra: List[Spectrum],
                         columns_dict: Dict[str, str] = None,
                         progress_bars: bool = True):
    """Wrapper to create sqlite file containing spectrum information needed for MS2Query

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        tables are added. If the tables in this sqlite file already exist, they
        will be overwritten.
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
    """
    initialize_tables(sqlite_file_name, additional_metadata_columns_dict=columns_dict)
    fill_spectrum_data_table(sqlite_file_name, list_of_spectra, progress_bar=progress_bars)

    fill_inchikeys_table(sqlite_file_name, list_of_spectra,
                         progress_bars=progress_bars)


def initialize_tables(sqlite_file_name: str,
                      additional_metadata_columns_dict: Dict[str, str] = None):
    """Creates a new sqlite file, with the empty tables spectrum_data and incikeys

    On default the columns spectrum_id and metadata are
    created. The column spectrum_id will be the primary key.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists a new
        table is added. If the table with the name specified by table_name
        already exists, then this table is overwritten.
    additional_metadata_columns_dict:
        Dictionary with as keys columns that need to be added in addition to
        the default columns and as values the datatype. The defaults columns
        are spectrum_id, peaks, intensities and metadata. The additional
        columns should be the same names that are in the metadata dictionary,
        since these values will be automatically added in the function
        add_list_of_spectra_to_sqlite.
        Default = None results in the default columns.
    """
    # Initialize spectrum_data_table
    # Combine default columns and additional metadata columns
    default_columns_dict = {"spectrumid": "INTEGER",
                            "metadata": "TEXT"}
    if additional_metadata_columns_dict is None:
        additional_metadata_columns_dict = {}
    combined_columns_dict = {**default_columns_dict, **additional_metadata_columns_dict}

    initialize_spectrum_data_table = """
    DROP TABLE IF EXISTS 'spectrum_data';
    CREATE TABLE 'spectrum_data' (
    """
    # add all columns with the type specified in combined_columns_dict
    for column_header in combined_columns_dict:
        initialize_spectrum_data_table += column_header + " " \
                                + combined_columns_dict[column_header] + ",\n"
    initialize_spectrum_data_table += "PRIMARY KEY (spectrumid));"

    # Initialize inchikeys table
    initialize_inchikeys_table = """;
    DROP TABLE IF EXISTS 'inchikeys';
    CREATE TABLE 'inchikeys'(
        'inchikey' TEXT,
        spectrum_ids_belonging_to_inchikey14 TEXT,
        closest_related_inchikeys TEXT,
        PRIMARY KEY ('inchikey')
        );
    """

    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()
    cur.executescript(initialize_spectrum_data_table)
    cur.executescript(initialize_inchikeys_table)
    conn.commit()
    conn.close()


def fill_spectrum_data_table(sqlite_file_name: str,
                             list_of_spectra: List[Spectrum],
                             progress_bar: bool = True):
    """Adds the data of a list of spectra to a sqlite file

    The spectrum_id is the primary key and the data is stored in metadata. If additional columns are specified in
    the table, the values stored for this column name in metadata are stored in the table.

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file to which the spectrum data is added. The sqlite
        file is expected to already contain a table with the name specified by
        table_name.
    list_of_spectra:
        The Spectra of which the data is added to the sqlite file.
    progress_bar:
        If True a progress bar will show the progress.
    """
    conn = sqlite3.connect(sqlite_file_name)

    # Get the column names in the sqlite file
    cur = conn.execute("select * from 'spectrum_data'")
    column_names = list(map(lambda x: x[0], cur.description))

    # Create a string with placeholders for the add spectrum command
    value_placeholders = ""
    for column in column_names:
        value_placeholders += ':' + column + ", "

    add_spectrum_command = f"""INSERT INTO 'spectrum_data'
                           values ({value_placeholders[:-2]})"""

    # Add the data of each spectrum to the sqlite table
    for i, spectrum in tqdm(enumerate(list_of_spectra),
                            desc="Adding spectra to sqlite table",
                            disable=not progress_bar):
        metadata = spectrum.metadata
        spectrumid = i

        # Give the value for the default columns
        value_dict = {"metadata": str(metadata),
                      "spectrumid": spectrumid}

        # Gets the data for addition columns from metadata
        for column in column_names:
            if column not in ['peaks', 'intensities', 'metadata', 'spectrumid']:
                if column in metadata:
                    value_dict[column] = str(metadata[column])
                else:
                    value_dict[column] = ""

        cur = conn.cursor()
        cur.execute(add_spectrum_command, value_dict)

    conn.commit()
    conn.close()


def fill_inchikeys_table(sqlite_file_name: str,
                         list_of_spectra: List[Spectrum],
                         progress_bars: bool = True):
    """Fills the inchikeys table with Inchikeys, spectrum_ids_belonging_to_inchikey and closest related inchikeys

    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        tables are added. If the tables in this sqlite file already exist, they
        will be overwritten.
    list_of_spectra:
        List of spectrum objects
    progress_bars:
        If True progress bars will show the progress of the different steps
        in the process.
    """
    # Get spectra belonging to each inchikey14
    spectra_belonging_to_inchikey14 = \
        get_spectra_belonging_to_inchikey14(list_of_spectra)

    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()

    closest_related_inchikey14s = calculate_highest_tanimoto_score(list_of_spectra, list_of_spectra, 10)

    # Fill table
    for inchikey14 in tqdm(spectra_belonging_to_inchikey14,
                           desc="Adding inchikey14s to sqlite table",
                           disable=not progress_bars):
        matching_spectrum_ids = str(spectra_belonging_to_inchikey14[inchikey14])
        add_row_to_table_command = \
            f"""INSERT INTO 'inchikeys' 
            values ("{inchikey14}", 
            "{matching_spectrum_ids}",
            "{str(closest_related_inchikey14s[inchikey14])}");"""
        cur.execute(add_row_to_table_command)
    conn.commit()
    conn.close()


def get_spectra_belonging_to_inchikey14(spectra: List[Spectrum]
                                        ) -> Dict[str, List[int]]:
    """Returns a dictionary with the spectrum_ids belonging to each inchikey14

    Args:
    -----
    inchikey14s:
        List of inchikey14s
    spectra:
        List of spectrum objects
    """
    spectra_belonging_to_inchikey14 = {}
    for spectrum_id, spectrum in enumerate(spectra):
        inchikey14_of_spectrum = spectrum.get("inchikey")[:14]
        if inchikey14_of_spectrum in spectra_belonging_to_inchikey14:
            spectra_belonging_to_inchikey14[inchikey14_of_spectrum].append(spectrum_id)
        else:
            spectra_belonging_to_inchikey14[inchikey14_of_spectrum] = [spectrum_id]
    return spectra_belonging_to_inchikey14
