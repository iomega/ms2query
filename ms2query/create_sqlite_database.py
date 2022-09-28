import sqlite3
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from matchms import Spectrum
from matchms import calculate_scores
from matchms.filtering import add_fingerprint
from matchms.similarity import FingerprintSimilarity
from tqdm import tqdm


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

    closest_related_inchikey14s = calculate_closest_related_inchikeys(list_of_spectra)

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


def select_inchi_for_unique_inchikeys(list_of_spectra: List[Spectrum]) -> (List[Spectrum], List[str]):
    """"Select spectra with most frequent inchi for unique inchikeys

    Method needed to calculate tanimoto scores"""
    # Select all inchi's and inchikeys from spectra metadata
    inchikeys_list = []
    inchi_list = []
    for s in list_of_spectra:
        inchikeys_list.append(s.get("inchikey"))
        inchi_list.append(s.get("inchi"))
    inchi_array = np.array(inchi_list)
    inchikeys14_array = np.array([x[:14] for x in inchikeys_list])

    # Select unique inchikeys
    inchikeys14_unique = list({x[:14] for x in inchikeys_list})

    spectra_with_most_frequent_inchi_per_unique_inchikey = []
    for inchikey14 in inchikeys14_unique:
        # Select inchis for inchikey14
        idx = np.where(inchikeys14_array == inchikey14)[0]
        inchis_for_inchikey14 = [list_of_spectra[i].get("inchi") for i in idx]
        # Select the most frequent inchi per inchikey
        inchi = Counter(inchis_for_inchikey14).most_common(1)[0][0]
        # Store the ID of the spectrum with the most frequent inchi
        ID = idx[np.where(inchi_array[idx] == inchi)[0][0]]
        spectra_with_most_frequent_inchi_per_unique_inchikey.append(list_of_spectra[ID].clone())
    return spectra_with_most_frequent_inchi_per_unique_inchikey, inchikeys14_unique


def calculate_closest_related_inchikeys(list_of_spectra: List[Spectrum]) -> Dict[str, List[Tuple[str, float]]]:
    spectra_with_most_frequent_inchi_per_inchikey, inchikeys14_unique = select_inchi_for_unique_inchikeys(list_of_spectra)
    # Add fingerprints
    fingerprint_spectra = []
    for spectrum in tqdm(spectra_with_most_frequent_inchi_per_inchikey,
                         desc="Calculating fingerprints for tanimoto scores"):
        spectrum_with_fingerprint = add_fingerprint(spectrum,
                                                    fingerprint_type="daylight",
                                                    nbits=2048)
        fingerprint_spectra.append(spectrum_with_fingerprint)

        assert spectrum_with_fingerprint.get("fingerprint") is not None, \
            f"Fingerprint for 1 spectrum could not be set smiles is {spectrum.get('smiles')}, inchi is {spectrum.get('inchi')}"

    # Specify type and calculate similarities
    similarity_measure = FingerprintSimilarity("jaccard")
    closest_related_inchikeys_dict = {}
    for fingerprint_spectrum in tqdm(fingerprint_spectra,
                                     desc="Calculating Tanimoto scores"):
        scores = calculate_scores([fingerprint_spectrum], fingerprint_spectra,
                                  similarity_measure,
                                  is_symmetric=False).scores[0]
        index_highest_scores = np.argpartition(scores, -10)[-10:]
        sorted_index_highest_scores = np.flip(index_highest_scores[np.argsort(scores[index_highest_scores])])
        inchikey_and_highest_scores = [(inchikeys14_unique[i], scores[i]) for i in sorted_index_highest_scores]

        closest_related_inchikeys_dict[fingerprint_spectrum.get("inchikey")[:14]] = inchikey_and_highest_scores
    return closest_related_inchikeys_dict
