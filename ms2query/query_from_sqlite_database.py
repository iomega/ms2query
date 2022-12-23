"""
Functions to obtain data from sqlite files.
"""

import ast
import sqlite3
from typing import Dict, List, Tuple


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
        f"""SELECT {spectrum_id_storage_name}, metadata FROM {table_name} 
        WHERE {spectrum_id_storage_name} 
        IN ('{"', '".join(map(str, spectrum_id_list))}')"""
    cur = conn.cursor()
    cur.execute(sqlite_command)
    list_of_metadata = cur.fetchall()
    # Convert to dictionary
    results_dict = {}
    for spectrumid, metadata in list_of_metadata:
        metadata = ast.literal_eval(metadata)
        results_dict[spectrumid] = metadata
    # Check if all spectrum_ids were found
    for spectrum_id in spectrum_id_list:
        assert spectrum_id in results_dict, \
            f"{spectrum_id_storage_name} {spectrum_id} not found in database"
    return results_dict


def get_ionization_mode_library(sqlite_file_name: str):
    conn = sqlite3.connect(sqlite_file_name)
    sqlite_command = "SELECT metadata FROM spectrum_data"
    cur = conn.cursor()
    cur.execute(sqlite_command)
    while True:
        metadata = cur.fetchone()
        # If all values have been checked None is returned.
        if metadata is None:
            print("The ionization mode of the library could not be determined")
            return None
        metadata = ast.literal_eval(metadata[0])
        if "ionmode" in metadata:
            ionmode = metadata["ionmode"]
            if ionmode == "positive":
                return "positive"
            if ionmode == "negative":
                return "negative"


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
