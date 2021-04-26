import json
import pickle
from typing import Dict, List, TextIO
from matchms.importing.load_from_json import as_spectrum
from matchms.Spectrum import Spectrum


def json_loader(file: TextIO) -> List[Spectrum]:
    """Read open json file with spectra and return

    Args:
    -------
    file:
        Open file of json to import.
    """
    spectrums = json.load(file)
    if not isinstance(spectrums, list):
        spectrums = [spectrums]
    for i, spec_i in enumerate(spectrums):
        spectrum = as_spectrum(spec_i)
        if spectrum is not None:
            spectrums[i] = spectrum

    return spectrums


def csv2dict(file_path: str) -> Dict[str, List[str]]:
    """Read csv into {first_column: [other_columns]}

    Args:
    -------
    file_path
        Filename of csv to read
    """
    csv_dict = {}
    with open(file_path, "r") as inf:
        for line in inf:
            line = line.strip()
            line = line.split(",")
            csv_dict[line[0]] = line[1:]
    return csv_dict


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object
