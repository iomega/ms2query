from typing import List
from matchms import Spectrum
import numpy as np
import random

def select_unique_inchikeys(spectra: List[Spectrum]) -> List[str]:
    """Selects unique inchikeys for a list of spectra"""
    inchikey_list = []
    for spectrum in spectra:
        inchikey = spectrum.get("inchikey")[:14]
        inchikey_list.append(inchikey)
    inchikey_set = set(inchikey_list)
    return sorted(list(inchikey_set))


def split_using_shuffle_split(unique_inchikeys: List[str],
                              k: int):
    from sklearn.model_selection import ShuffleSplit
    rs = ShuffleSplit(n_splits=k, test_size=1/k)
    for train_index, test_index in rs.split(unique_inchikeys):
        print("TRAIN:", train_index, "TEST:", test_index)


def split_spectra_in_5_sets(unique_inchikeys: List[str],
                            k: int) -> List[List[str]]:
    """Splits a set of inchikeys into 5 sets with the same number of inchikeys"""
    random.shuffle(unique_inchikeys)
    inchikey_set_size = len(unique_inchikeys)//k
    inchikey_sets = []
    for i in range(k):
        inchikey_sets.append(unique_inchikeys[inchikey_set_size*i:inchikey_set_size*(i+1)])

    # The length of the number of spectra might not be devidable by k. So a few inchikeys do not end up somewhere.
    number_of_missing_inchikeys = len(unique_inchikeys)%k

    for

    return inchikey_sets


if __name__ == "__main__":


    # load in cleaned positive mode spectra.
    # Select all unique inchikeys
    # Split in 5, select matching spectra.
    # Train MS2Deepscore and Spec2Vec on 4/5th training spectra
    # Train MS2Query with MS2Deepscore and Spec2Vec
    # Use test set on MS2Query to test performance
    # Store the complete model and the 5 data splits.