"""
This script is not needed for normally running MS2Query, it is only needed to generate a new library or to train
new models
"""

from typing import List, Dict
from matchms import Spectrum
import random


def select_unique_inchikeys(spectra: List[Spectrum]) -> List[str]:
    """Selects unique inchikeys for a list of spectra"""
    inchikey_list = []
    for spectrum in spectra:
        inchikey = spectrum.get("inchikey")[:14]
        inchikey_list.append(inchikey)
    inchikey_set = set(inchikey_list)
    return sorted(list(inchikey_set))


def split_spectra_in_random_inchikey_sets(spectra: List[Spectrum],
                                          k: int) -> List[List[Spectrum]]:
    """Splits a set of inchikeys into k sets with the same number of inchikeys"""
    spectrum_sets = []
    unique_inchikeys = select_unique_inchikeys(spectra)
    random.shuffle(unique_inchikeys)
    fraction_size = len(unique_inchikeys) // k
    for _ in range(k):
        validation_inchikeys = unique_inchikeys[-fraction_size:]
        unique_inchikeys = unique_inchikeys[:-fraction_size]
        spectrum_set, spectra = select_spectra_belonging_to_inchikey(spectra, validation_inchikeys)
        spectrum_sets.append(spectrum_set)

    # Devide left over inchikeys over sets.
    # If the number of inchikeys is not perfectly dividable, some inchikeys will be left over, these are added
    for set_nr, inchikey in enumerate(unique_inchikeys):
        spectrum_set, spectra = select_spectra_belonging_to_inchikey(spectra, [inchikey])
        spectrum_sets[set_nr] += spectrum_set
    return spectrum_sets


def select_spectra_per_unique_inchikey(spectra: List[Spectrum]) -> Dict[str, List[Spectrum]]:
    inchikey_dict = {}
    for spectrum in spectra:
        inchikey = spectrum.get("inchikey")[:14]
        if inchikey in inchikey_dict:
            inchikey_dict[inchikey].append(spectrum)
        else:
            inchikey_dict[inchikey] = [spectrum]
    return inchikey_dict


def select_spectra_belonging_to_inchikey(spectra: List[Spectrum],
                                         inchikeys: List[str]) -> (List[Spectrum], List[Spectrum]):
    # Select spectra belonging to the selected inchikeys
    spectra_containing_inchikey = []
    spectra_not_containing_inchikey = []
    for spectrum in spectra:
        inchikey = spectrum.get("inchikey")[:14]
        if inchikey in inchikeys:
            spectra_containing_inchikey.append(spectrum)
        else:
            spectra_not_containing_inchikey.append(spectrum)
    return spectra_containing_inchikey, spectra_not_containing_inchikey


def split_spectra_on_inchikeys(spectra, validation_fraction):
    """Splits a set of inchikeys and selects all spectra beloning to these inchikeys"""
    # Select unique inchikeys and select a random set to be the validation inchikeys
    unique_inchikeys = select_unique_inchikeys(spectra)
    random.shuffle(unique_inchikeys)
    nr_of_inchikeys = len(unique_inchikeys)//validation_fraction
    validation_inchikeys = unique_inchikeys[-nr_of_inchikeys:]

    # Select spectra belonging to the selected inchikeys
    validation_spectra, training_spectra = select_spectra_belonging_to_inchikey(spectra, validation_inchikeys)
    return training_spectra, validation_spectra


def split_training_and_validation_spectra(spectra: List[Spectrum], validation_fraction=10):
    """Randomly selects a validation set from test spectra

    ratio:
        The fraction of the spectra that should be training spectra
    :returns
        Training spectra, validation spectra"""
    random.shuffle(spectra)
    training_spectra = spectra[:-len(spectra)//validation_fraction]
    validation_spectra = spectra[-len(spectra)//validation_fraction:]
    return training_spectra, validation_spectra
