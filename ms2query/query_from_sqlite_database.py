"""
Functions to obtain data from sqlite files.
"""

import ast
import os.path
import sqlite3
from typing import Dict, List, Tuple


class SqliteLibrary:
    def __init__(self,
                 sqlite_file_name,
                 spectrum_id_storage_name="spectrumid"):
        assert os.path.isfile(sqlite_file_name), "The given sqlite file does not exist"
        self.sqlite_file_name = sqlite_file_name
        self.spectrum_id_storage_name = spectrum_id_storage_name


    def get_metadata_from_sqlite(self,
                                 spectrum_id_list: List[int],
                                 spectrum_id_storage_name: str = "spectrumid",
                                 table_name: str = "spectrum_data"
                                 ) -> Dict[int, dict]:
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
        conn = sqlite3.connect(self.sqlite_file_name)
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

    def get_ionization_mode_library(self):
        conn = sqlite3.connect(self.sqlite_file_name)
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

    def get_precursor_mz(self,
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
        conn = sqlite3.connect(self.sqlite_file_name)
        sqlite_command = \
            f"SELECT {self.spectrum_id_storage_name}, precursor_mz FROM spectrum_data"
        cur = conn.cursor()
        cur.execute(sqlite_command)
        results = cur.fetchall()
        precursor_mz_dict = {}
        for result in results:
            precursor_mz_dict[result[0]] = result[1]
        return precursor_mz_dict


    def get_inchikey_information(self) -> Tuple[Dict[str, List[str]],
                                            Dict[str, List[Tuple[str, float]]]]:
        """Returns the closely related inchikeys and the matching spectrum ids

        sqlite_file_name:
            The file name of an sqlite file
        """
        conn = sqlite3.connect(self.sqlite_file_name)
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


    def get_classes_inchikeys(self,
                              inchikeys):
        conn = sqlite3.connect(self.sqlite_file_name)
        sqlite_command = f"""SELECT 'smiles', 'cf_kingdom',
            'cf_superclass', 'cf_class', 'cf_subclass', 'cf_direct_parent',
            'npc_class_results', 'npc_superclass_results', 'npc_pathway_results',
            'npc_isglycoside' FROM inchikeys WHERE inchikey IN ('{"', '".join(map(str, inchikeys))}')"""
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