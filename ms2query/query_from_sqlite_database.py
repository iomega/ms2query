import sqlite3
from typing import Dict, List
from matchms.Spectrum import Spectrum
import ast
from matchms.importing.load_from_json import dict2spectrum


def get_spectra_from_sqlite(sqlite_file_name: str,
                            spectrum_id_list: List[str],
                            table_name: str = "spectra") -> List[Spectrum]:
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
    conn = sqlite3.connect(sqlite_file_name)

    sqlite_command = f"""SELECT full_json FROM {table_name} 
                    WHERE spectrum_id 
                    IN ('{"', '".join(map(str, spectrum_id_list))}')"""

    cur = conn.cursor()
    cur.execute(sqlite_command)

    list_of_spectra_dict = []
    for json_spectrum in cur:
        # Remove the "()" around the spectrum
        json_spectrum = json_spectrum[0]
        # Convert string to dictionary
        json_spectrum = ast.literal_eval(json_spectrum)
        list_of_spectra_dict.append(json_spectrum)
    conn.close()

    # Convert to matchms.Spectrum.Spectrum object
    spectra_list = []
    for spectrum in list_of_spectra_dict:
        spectra_list.append(dict2spectrum(spectrum))
    return spectra_list


def get_tanimoto_from_sqlite(sqlite_file_name: str):
    conn = sqlite3.connect(sqlite_file_name)

    sqlite_command = """SELECT inchikey1, inchikey2, tanimoto_score FROM tanimoto 
                    WHERE inchikey1 in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,200) and inchikey2 in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,200);
                    """

    cur = conn.cursor()
    cur.execute(sqlite_command)
    for score in cur:
        print(score)
    conn.close()

    # return tanimoto_score

