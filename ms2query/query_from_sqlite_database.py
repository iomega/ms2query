"""
Functions to obtain data from sqlite files.
"""

import ast
import io
import sqlite3
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from matchms.Spectrum import Spectrum
from tqdm import tqdm


def get_spectra_from_sqlite(sqlite_file_name: str,
                            spectrum_id_list: List[str],
                            spectrum_id_storage_name: str = "spectrumid",
                            table_name: str = "spectrum_data",
                            get_all_spectra: bool = False,
                            progress_bar: bool = False
                            ) -> List[Spectrum]:
    """Returns a list with all spectra specified in spectrum_id_list

    Args:
    -------
    sqlite_file_name:
        File name of the sqlite file that contains the spectrum information
    spectrum_id_list:
        List of spectrum_id's of which the spectra objects should be returned
    spectrum_id_storage_name:
        The name under which the spectrum ids are stored in the metadata.
        Default = 'spectrumid'
    table_name:
        Name of the table in the sqlite file that stores the spectrum data
    get_all_spectra:
        When True all spectra in the sqlite table are returned, instead of
        only the once mentioned in spectrum_id_list, in this case
        spectrum_id_list can be an empty list. Default = False.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    # Converts TEXT to np.array when querying
    sqlite3.register_converter("array", convert_array)

    conn = sqlite3.connect(sqlite_file_name,
                           detect_types=sqlite3.PARSE_DECLTYPES)

    # Get all relevant data.
    sqlite_command = f"SELECT peaks, intensities, metadata FROM {table_name} "
    if not get_all_spectra:
        sqlite_command += f"""WHERE {spectrum_id_storage_name} 
                          IN ('{"', '".join(map(str, spectrum_id_list))}')"""
    cur = conn.cursor()
    cur.execute(sqlite_command)
    list_of_results = cur.fetchall()
    # Convert to list of matchms.Spectrum
    spectra_dict = {}
    for result in tqdm(list_of_results,
                       desc="Converting to Spectrum objects",
                       disable=not progress_bar):
        peaks = result[0]
        intensities = result[1]
        metadata = ast.literal_eval(result[2])
        spectrum_id = metadata[spectrum_id_storage_name]
        spectra_dict[spectrum_id] = Spectrum(mz=peaks,
                                             intensities=intensities,
                                             metadata=metadata)
    conn.close()
    if get_all_spectra:
        list_of_spectra = list(spectra_dict.values())
    else:
        # Make sure the returned list has same order as spectrum_id_list
        list_of_spectra = []
        for spectrum_id in spectrum_id_list:
            assert spectrum_id in spectra_dict, \
                f"No spectrum with spectrum_id: {spectrum_id} " \
                f"was found in sqlite"
            list_of_spectra.append(spectra_dict[spectrum_id])
    return list_of_spectra


def convert_array(nd_array_in_bytes: bytes) -> np.array:
    """Converts np.ndarray stored in binary format back to an np.ndarray

    By running this command:
    sqlite3.register_converter("array", convert_array)
    This function will be called everytime something with the datatype array is
    loaded from a sqlite file.
    Found at: http://stackoverflow.com/a/31312102/190597 (SoulNibbler)

    Args:
    -------
    nd_array_in_bytes:
        A numpy array stored in a binary format.
    """
    out = io.BytesIO(nd_array_in_bytes)
    out.seek(0)
    return np.load(out)


def get_tanimoto_score_for_inchikey14s(list_of_inchikey14s_1: List[str],
                                       list_of_inchikey14s_2: List[str],
                                       sqlite_file_name: str) -> pd.DataFrame:
    """Returns a panda dataframe with the tanimoto scores

    Args:
    -------
    list_of_identifiers_1:
        A list with inchikey14s, the tanimoto scores between this list and
        list_of_identifiers_2 are returned.
    list_of_identifiers_2:
        A list with inchikey14s, the tanimoto scores between this list and
        list_of_identifiers_1 are returned
    sqlite_file_name:
        The sqlite file in which the tanimoto scores are stored.
    """
    # Gets the identifiers from the inchikey14s
    inchikey14_dict_1 = get_index_of_inchikey14s(list_of_inchikey14s_1,
                                                 sqlite_file_name)
    inchikey14_dict_2 = get_index_of_inchikey14s(list_of_inchikey14s_2,
                                                 sqlite_file_name)
    identifiers_list_1 = list(inchikey14_dict_1.values())
    identifiers_list_2 = list(inchikey14_dict_2.values())

    # Get the tanimoto scores between the inchikey14s in list_of_inchikey14s
    tanimoto_score_matrix = get_tanimoto_from_sqlite(sqlite_file_name,
                                                     identifiers_list_1,
                                                     identifiers_list_2)
    # Reverse the dictionary value becomes key and key becomes value
    reversed_inchikey14_dict_1 = {v: k for k, v in inchikey14_dict_1.items()}
    reversed_inchikey14_dict_2 = {v: k for k, v in inchikey14_dict_2.items()}

    # Change the column names and indexes from identifiers to inchikey14s
    tanimoto_score_matrix.rename(
        columns=reversed_inchikey14_dict_2,
        index=reversed_inchikey14_dict_1, inplace=True)
    return tanimoto_score_matrix


def get_index_of_inchikey14s(list_of_inchikey14s: List[str],
                             sqlite_file_name: str,
                             table_name: str = "inchikeys") -> Dict[str, int]:
    """Look up the identifiers for each inchikey14 in a sqlite file.

    Args:
    -------
    list_of_inchikey14s:
        A list of 14 letter inchikeys
    sqlite_file_name:
        The file name of a sqlite file in which the identifiers of the
        inchikey14 are stored
    table_name:
        The table name in which the identifiers or the inchikey14 are stored.
        Default = "inchikeys"
    """

    conn = sqlite3.connect(sqlite_file_name)

    # Create string with placeholders for sqlite command
    question_marks = ''
    for _ in range(len(list_of_inchikey14s)):
        question_marks += '?,'
    # Remove the last comma
    question_marks = question_marks[:-1]

    sqlite_command = f"""SELECT inchikey, rowid FROM {table_name}
                        WHERE inchikey in ({question_marks})"""

    cur = conn.cursor()
    cur.execute(sqlite_command, list_of_inchikey14s)
    # Convert result to dictionary
    identifier_dict = {}
    for result in cur:
        identifier_dict[result[0]] = result[1]

    conn.close()

    # Check if all inchikey14s are found
    for inchikey14 in list_of_inchikey14s:
        assert inchikey14 in identifier_dict, \
            f"Inchikey14 {inchikey14} was not found in sqlite file"
    return identifier_dict


def get_tanimoto_from_sqlite(sqlite_file_name: str,
                             list_of_identifiers_1: List[int],
                             list_of_identifiers_2) -> pd.DataFrame:
    """Returns the tanimoto scores between the lists of identifiers

    args:
    ------
    sqlite_file_name:
        The sqlite file in which the tanimoto scores are stored.
    list_of_identifiers_1:
        A list with inchikey14s, the tanimoto scores between this list and
        list_of_identifiers_2 are returned.
    list_of_identifiers_2:
        A list with inchikey14s, the tanimoto scores between this list and
        list_of_identifiers_1 are returned
    """
    # pylint: disable=too-many-locals
    conn = sqlite3.connect(sqlite_file_name)
    identifier_string_1 = ",".join([str(x) for x in list_of_identifiers_1])
    identifier_string_2 = ",".join([str(x) for x in list_of_identifiers_2])

    sqlite_command = f"""SELECT identifier_1, identifier_2, tanimoto_score
                    FROM tanimoto_scores
                    WHERE identifier_1 in ({identifier_string_1})
                    and identifier_2 in ({identifier_string_2})
                    UNION
                    SELECT identifier_1, identifier_2, tanimoto_score
                    FROM tanimoto_scores
                    WHERE identifier_1 in ({identifier_string_2})
                    and identifier_2 in ({identifier_string_1})
                    ;
                    """
    cur = conn.cursor()
    cur.execute(sqlite_command)
    results = cur.fetchall()
    result_dict = {}
    for result in results:
        identifier_1 = result[0]
        identifier_2 = result[1]
        tanimoto_score = result[2]
        # Store the tanimoto score twice also with reversed combination
        result_dict[(identifier_1, identifier_2)] = tanimoto_score
        result_dict[(identifier_2, identifier_1)] = tanimoto_score
    # Create matrix with tanimoto scores
    tanimoto_scores = []
    for identifier_1 in list_of_identifiers_1:
        current_row = []
        for identifier_2 in list_of_identifiers_2:
            tanimoto_score = result_dict[(identifier_1, identifier_2)]
            current_row.append(tanimoto_score)
        tanimoto_scores.append(current_row)
    # Store as pd.dataframe
    result_dataframe = pd.DataFrame(data=tanimoto_scores,
                                    index=list_of_identifiers_1,
                                    columns=list_of_identifiers_2)
    conn.close()
    return result_dataframe


def get_metadata_from_sqlite(sqlite_file_name: str,
                             spectrum_id_list: List[str],
                             spectrum_id_storage_name: str = "spectrumid",
                             table_name: str = "spectrum_data"
                             ) -> Dict[str, dict]:
    """Returns a dict with as values the metadata for each spectrum id

    Args:
    ------
    sqlite_file_name:
        The sqlite file in which the spectra data is stored.
    spectrum_id_list:
        A list with spectrum ids for which the part of the metadata should be
        looked up.
    spectrum_id_storage_name:
        The name under which the spectrum ids are stored in the metadata.
        Default = 'spectrumid'
    table_name:
        The name of the table in the sqlite file in which the metadata is
        stored. Default = "spectrum_data"
    """
    conn = sqlite3.connect(sqlite_file_name)
    sqlite_command = \
        f"""SELECT metadata FROM {table_name} 
        WHERE {spectrum_id_storage_name} 
        IN ('{"', '".join(map(str, spectrum_id_list))}')"""
    cur = conn.cursor()
    cur.execute(sqlite_command)
    list_of_metadata = cur.fetchall()
    # Convert to dictionary
    results_dict = {}
    for metadata in list_of_metadata:
        metadata = ast.literal_eval(metadata[0])
        results_dict[metadata[spectrum_id_storage_name]] = metadata
    # Check if all spectrum_ids were found
    for spectrum_id in spectrum_id_list:
        assert spectrum_id in results_dict, \
            f"{spectrum_id_storage_name} {spectrum_id} not found in database"
    return results_dict


def get_precursor_mz_within_range(sqlite_file_name: str,
                                 lower_bound: Union[float, int],
                                 upper_bound: Union[float, int],
                                 spectrum_id_storage_name: str = "spectrumid",
                                 table_name: str = "spectrum_data"
                                 ) -> List[Tuple[str, float]]:
    """Returns spectrum_ids with precursor m/z between lower and upper bound

    Args:
    -----
    sqlite_file_name:
        The sqlite file in which the spectra data is stored.
    lower_bound:
        The lower bound of the allowed precursor m/z
    upper_bound:
        The upper bound of the allowed precursor m/z
    spectrum_id_storage_name:
        The name under which the spectrum ids are stored in the metadata.
        Default = 'spectrumid'
    table_name:
        The name of the table in the sqlite file in which the metadata is
        stored. Default = "spectrum_data"
    """
    conn = sqlite3.connect(sqlite_file_name)
    sqlite_command = \
        f"""SELECT {spectrum_id_storage_name}, precursor_mz FROM {table_name} 
        WHERE precursor_mz BETWEEN {lower_bound} and {upper_bound}"""
    cur = conn.cursor()
    cur.execute(sqlite_command)
    spectrum_ids_within_range = cur.fetchall()
    return spectrum_ids_within_range


def get_precursor_mz(sqlite_file_name: str,
                    spectrum_id_storage_name: str = "spectrumid",
                    table_name: str = "spectrum_data"
                    ) -> Dict[str, float]:
    """Returns all spectrum_ids with precursor m/z

    Args:
    -----
    sqlite_file_name:
        The sqlite file in which the spectra data is stored.
    spectrum_id_storage_name:
        The name under which the spectrum ids are stored in the metadata.
        Default = 'spectrumid'
    table_name:
        The name of the table in the sqlite file in which the metadata is
        stored. Default = "spectrum_data"
    """
    conn = sqlite3.connect(sqlite_file_name)
    sqlite_command = \
        f"SELECT {spectrum_id_storage_name}, precursor_mz FROM {table_name}"
    cur = conn.cursor()
    cur.execute(sqlite_command)
    results = cur.fetchall()
    precursor_mz_dict = {}
    for result in results:
        precursor_mz_dict[result[0]] = result[1]
    return precursor_mz_dict


def get_inchikey_information(sqlite_file_name: str
                             ) -> Tuple[Dict[str, List[str]],
                                        Dict[str, List[Tuple[str, float]]]]:
    """Returns the closely related inchikeys and the matching spectrum ids

    sqlite_file_name:
        The file name of an sqlite file
    """
    # todo add test function
    conn = sqlite3.connect(sqlite_file_name)
    sqlite_command = "SELECT * FROM inchikeys"
    cur = conn.cursor()
    cur.execute(sqlite_command)
    results = cur.fetchall()
    matching_spectrum_ids_dict = {}
    closely_related_inchikeys_dict = {}
    for row in results:
        inchikey = row[0]
        matching_spectrum_ids = ast.literal_eval(row[1])
        closely_related_inchikeys = ast.literal_eval(row[2])
        matching_spectrum_ids_dict[inchikey] = matching_spectrum_ids
        closely_related_inchikeys_dict[inchikey] = closely_related_inchikeys
    return matching_spectrum_ids_dict, closely_related_inchikeys_dict
