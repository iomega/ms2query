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
        sqlite_command += f"""WHERE spectrum_id 
                          IN ('{"', '".join(map(str, spectrum_id_list))}')"""
    cur = conn.cursor()
    cur.execute(sqlite_command)

    # Convert to list of matchms.Spectrum
    list_of_spectra = []
    for result in tqdm(cur,
                       desc="Loading spectra from sqlite",
                       disable=not progress_bar):
        peaks = result[0]
        intensities = result[1]
        metadata = ast.literal_eval(result[2])

        list_of_spectra.append(Spectrum(mz=peaks,
                                        intensities=intensities,
                                        metadata=metadata))
    conn.close()
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


def get_tanimoto_score_for_inchikeys(list_of_inchikeys: List[str],
                                     sqlite_file_name: str) -> pd.DataFrame:
    """Returns a panda dataframe matrix with the tanimoto scores

    Args:
    -------
    list_of_inchikeys:
        A list with all the inchikeys of between which the tanimoto scores
        are returned.
    sqlite_file_name:
        The sqlite file in which the tanimoto scores are stored.
    """
    # Gets the identifiers from the inchikeys
    inchikey_dict = get_index_of_inchikeys(list_of_inchikeys, sqlite_file_name)
    identifier_list = []
    for inchikey in inchikey_dict:
        identifier_list.append(inchikey_dict[inchikey])

    # Get the tanimoto scores between the inchikeys in list_of_inchikeys
    tanimoto_score_matrix = get_tanimoto_from_sqlite(sqlite_file_name,
                                                     identifier_list)
    # Reverse the dictionary value becomes key and key becomes value
    reversed_inchikey_dict = {v: k for k, v in inchikey_dict.items()}
    # Change the column names and indexes from identifiers to inchikeys
    tanimoto_score_matrix.rename(columns=reversed_inchikey_dict,
                                 index=reversed_inchikey_dict, inplace=True)
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
        identifier_dict[result[0]] = result[1]-1

    conn.close()

    # Check if all inchikeys are found
    for inchikey in list_of_inchikeys:
        if inchikey not in identifier_dict:
            print(inchikey + " is not found")
    return identifier_dict


def get_tanimoto_from_sqlite(sqlite_file_name: str,
                             list_of_identifiers: List[int]) -> pd.DataFrame:
    """Returns the tanimoto scores between the identifiers

    args:
    ------
    sqlite_file_name:
        The sqlite file in which the tanimoto scores are stored.
    list_of_identifiers:
        A list with all the inchikeys of between which the tanimoto scores
        are returned.
    """
    conn = sqlite3.connect(sqlite_file_name)
    identifier_string = ",".join([str(x) for x in list_of_identifiers])

    sqlite_command = f"""SELECT identifier_1, identifier_2, tanimoto_score 
                    FROM tanimoto_scores
                    WHERE identifier_1 in ({identifier_string}) 
                    and identifier_2 in ({identifier_string});
                    """
    cur = conn.cursor()
    cur.execute(sqlite_command)
    result_list = []
    for result in cur:
        result_list.append(result)

    # The data is changed to pd.DataFrame twice and then added together. So
    # that both the tanimoto score is given for 1,2 and 2,1. Thereby
    # duplicating the data, but making the lookup easier for other functions.
    scores_normal_identifiers = pd.DataFrame(result_list,
                                             columns=["identifier_1",
                                                      "identifier_2",
                                                      "tanimoto_score"])
    scores_reversed_identifiers = pd.DataFrame(result_list,
                                               columns=["identifier_2",
                                                        "identifier_1",
                                                        "tanimoto_score"])
    result_dataframe_melt_structure = pd.concat([scores_normal_identifiers,
                                                 scores_reversed_identifiers])

    # Changes the structure of the database from a melt structure to a matrix
    result_dataframe = pd.pivot_table(result_dataframe_melt_structure,
                                      columns="identifier_1",
                                      index="identifier_2",
                                      values="tanimoto_score")
    conn.close()
    return result_dataframe
