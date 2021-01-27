import sqlite3
from typing import Dict, List
from matchms.Spectrum import Spectrum
import ast
from matchms.importing.load_from_json import dict2spectrum
import time
import pandas as pd
import numpy as np
import io

def get_spectra_from_sqlite(sqlite_file_name: str,
                            spectrum_id_list: List[str],
                            table_name: str = "spectrum_data") -> List[Spectrum]:
    """Returns a list with all metadata of spectrum_ids in spectrum_id_list

    Args:
    -------
    sqlite_file_name:
        File name of the sqlite file that contains the spectrum information
    spectrum_id_list:
        List of spectrum_id's of which the metadata should be returned
    table_name:
        Name of the table in the sqlite file that stores the spectrum data
    """
    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)

    conn = sqlite3.connect(sqlite_file_name,
                           detect_types=sqlite3.PARSE_DECLTYPES)

    sqlite_command = f"""SELECT peaks, intensities, metadata FROM {table_name} 
                    WHERE spectrum_id 
                    IN ('{"', '".join(map(str, spectrum_id_list))}')"""

    cur = conn.cursor()
    cur.execute(sqlite_command)
    list_of_spectra = []
    for result in cur:
        peaks = result[0]
        intensities = result[1]
        metadata = ast.literal_eval(result[2])

        list_of_spectra.append(Spectrum(mz=peaks,
                            intensities=intensities,
                            metadata=metadata))
    conn.close()
    return list_of_spectra


def convert_array(text):
    out = io.BytesIO(text)
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
    inchikey_dict = get_index_of_inchikeys(list_of_inchikeys, sqlite_file_name)
    identifier_list = []
    for inchikey in inchikey_dict:
        identifier_list.append(inchikey_dict[inchikey])

    tanimoto_score_matrix = get_tanimoto_from_sqlite(sqlite_file_name,
                                                     identifier_list)
    # Change the column names and indexes from identifiers to inchikeys
    reversed_inchikey_dict = {v: k for k, v in inchikey_dict.items()}
    tanimoto_score_matrix.rename(columns=reversed_inchikey_dict,
                                 index=reversed_inchikey_dict, inplace=True)
    return tanimoto_score_matrix


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
    identifier_string = ""
    for identifier in list_of_identifiers:
        identifier_string += str(identifier) + ","
    # Remove last comma
    identifier_string = identifier_string[:-1]

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

    result_dataframe = pd.DataFrame(result_list,
                                    columns=["identifier_1",
                                             "identifier_2",
                                             "tanimoto_score"])
    result_dataframe = pd.pivot_table(result_dataframe,
                                      columns="identifier_1",
                                      index="identifier_2",
                                      values="tanimoto_score")
    conn.close()
    return result_dataframe


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
        The table name in which the identifiers or the inchikey are stored
    """

    conn = sqlite3.connect(sqlite_file_name)

    # Create string with placeholders for sqlite command
    question_marks = ''
    for i in range(len(list_of_inchikeys)):
        question_marks += '?,'
    # Remove the last comma
    question_marks = question_marks[:-1]

    sqlite_command = f"""SELECT inchikey, rowid FROM {table_name}
                        WHERE inchikey in ({question_marks})"""

    cur = conn.cursor()
    cur.execute(sqlite_command, list_of_inchikeys)
    # Conver result to dictionary
    identifier_dict = {}
    for result in cur:
        identifier_dict[result[0]] = result[1]-1

    conn.close()

    # Check if all inchikeys are found
    for inchikey in list_of_inchikeys:
        if inchikey not in identifier_dict:
            print(inchikey + " is not found")
    return identifier_dict


# get_spectra_from_sqlite("../downloads/data_all_inchikeys.sqlite", ["CCMSLIB00000001547", "CCMSLIB00000001548"])
# get_tanimoto_from_sqlite("../downloads/data_all_inchikeys.sqlite",[])
# print(get_tanimoto_score_for_inchikeys(['MYHSVHWQEVDFQT',
#                                         'HYTGGNIMZXFORS',
#                                         'JAMSDVDUWQNQFZ',
#                                         'QASOACWKTAXFSE',
#                                         'MZWQZYDNVSKPPF',
#                                         'abc'],
#                                        "../downloads/data_all_inchikeys.sqlite"))
# result = get_spectra_from_sqlite("../tests/test_spectra_database.sqlite", ['CCMSLIB00000001547',
#                                       'CCMSLIB00000001548',
#                                       'CCMSLIB00000001554'])
# my_array = np.array([1.2, 15.6, 12.3])

