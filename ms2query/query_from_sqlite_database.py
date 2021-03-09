"""
Functions to obtain data from sqlite files.
"""

import ast
import io
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlite3
from matchms.Spectrum import Spectrum


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
        print(result)
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
                f"No spectrum with spectrum_id: {spectrum_id} was found in sqlite"
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


def get_tanimoto_score_for_inchikeys(list_of_inchikeys_1: List[str],
                                     list_of_inchikeys_2: List[str],
                                     sqlite_file_name: str) -> pd.DataFrame:
    """Returns a panda dataframe with the tanimoto scores

    Args:
    -------
    list_of_identifiers_1:
        A list with inchikeys, the tanimoto scores between this list and
        list_of_identifiers_2 are returned.
    list_of_identifiers_2:
        A list with inchikeys, the tanimoto scores between this list and
        list_of_identifiers_1 are returned
    sqlite_file_name:
        The sqlite file in which the tanimoto scores are stored.
    """
    # Gets the identifiers from the inchikeys
    inchikey_dict_1 = get_index_of_inchikeys(list_of_inchikeys_1,
                                             sqlite_file_name)
    inchikey_dict_2 = get_index_of_inchikeys(list_of_inchikeys_2,
                                             sqlite_file_name)
    identifiers_list_1 = list(inchikey_dict_1.values())
    identifiers_list_2 = list(inchikey_dict_2.values())

    # Get the tanimoto scores between the inchikeys in list_of_inchikeys
    tanimoto_score_matrix = get_tanimoto_from_sqlite(sqlite_file_name,
                                                     identifiers_list_1,
                                                     identifiers_list_2)
    # Reverse the dictionary value becomes key and key becomes value
    reversed_inchikey_dict_1 = {v: k for k, v in inchikey_dict_1.items()}
    reversed_inchikey_dict_2 = {v: k for k, v in inchikey_dict_2.items()}

    # Change the column names and indexes from identifiers to inchikeys
    tanimoto_score_matrix.rename(columns=reversed_inchikey_dict_2,
                                 index=reversed_inchikey_dict_1, inplace=True)
    return tanimoto_score_matrix


def get_index_of_inchikeys(list_of_inchikeys: List[str],
                           sqlite_file_name: str,
                           table_name: str = "inchikeys") -> Dict[str, int]:
    """Look up the identifiers for each inchikey in a sqlite file.

    Args:
    -------
    list_of_inchikeys:
        A list of 14 letter inchikeys
    sqlite_file_name:
        The file name of a sqlite file in which the identifiers of the inchikey
        are stored
    table_name:
        The table name in which the identifiers or the inchikey are stored.
        Default = "inchikeys"
    """

    conn = sqlite3.connect(sqlite_file_name)

    # Create string with placeholders for sqlite command
    question_marks = ''
    for _ in range(len(list_of_inchikeys)):
        question_marks += '?,'
    # Remove the last comma
    question_marks = question_marks[:-1]

    sqlite_command = f"""SELECT inchikey, rowid FROM {table_name}
                        WHERE inchikey in ({question_marks})"""

    cur = conn.cursor()
    cur.execute(sqlite_command, list_of_inchikeys)
    # Convert result to dictionary
    identifier_dict = {}
    for result in cur:
        identifier_dict[result[0]] = result[1]

    conn.close()

    # Check if all inchikeys are found
    for inchikey in list_of_inchikeys:
        assert inchikey in identifier_dict, \
            f"Inchikey {inchikey} was not found in sqlite file"
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
        A list with inchikeys, the tanimoto scores between this list and
        list_of_identifiers_2 are returned.
    list_of_identifiers_2:
        A list with inchikeys, the tanimoto scores between this list and
        list_of_identifiers_1 are returned
    """
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
    for row, identifier_1 in enumerate(list_of_identifiers_1):
        current_row = []
        for column, identifier_2 in enumerate(list_of_identifiers_2):
            tanimoto_score = result_dict[(identifier_1, identifier_2)]
            current_row.append(tanimoto_score)
        tanimoto_scores.append(current_row)
    # Store as pd.dataframe
    result_dataframe = pd.DataFrame(data=tanimoto_scores,
                                    index=list_of_identifiers_1,
                                    columns=list_of_identifiers_2)
    conn.close()
    return result_dataframe


def get_part_of_metadata_from_sqlite(sqlite_file_name: str,
                                     spectrum_id_list: List[str],
                                     part_of_metadata_to_select: str,
                                     spectrum_id_storage_name: str
                                     = "spectrumid",
                                     table_name: str = "spectrum_data"
                                     ) -> List[str]:
    """Returns a dict with part of metadata for each spectrum id

    The key of the dict are the spectrum_ids and the values the part of the
    metadata that was marked with part_of_metadata_to_select.

    Args:
    ------
    sqlite_file_name:
        The sqlite file in which the spectra data is stored.
    spectrum_id_list:
        A list with spectrum ids for which the part of the metadata should be
        looked up.
    part_of_metadata_to_select:
        The key under which this metadata is stored in the sqlite file.
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
    results_dict = {}
    for metadata in list_of_metadata:
        metadata = ast.literal_eval(metadata[0])
        results_dict[metadata[spectrum_id_storage_name]] = \
            metadata[part_of_metadata_to_select]
    # Check if all spectrum_ids were found
    for spectrum_id in spectrum_id_list:
        assert spectrum_id in results_dict, \
            f"{spectrum_id_storage_name} {spectrum_id} not found in database"
    # Output from get_part_of_metadata is not always in order of input, so this
    # is sorted again here.
    results_in_correct_order = \
        [results_dict[spectrum_id]
         for spectrum_id in spectrum_id_list]
    return results_in_correct_order
