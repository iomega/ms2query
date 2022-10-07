from typing import List
from matchms import Spectrum


def select_unique_inchikeys(spectra: List[Spectrum]) -> List[str]:
    """Selects unique inchikeys for a list of spectra"""
    inchikey_list = []
    for spectrum in spectra:
        inchikey = spectrum.get("inchikey")[:14]
        inchikey_list.append(inchikey)
    inchikey_set = set(inchikey_list)
    return sorted(list(inchikey_set))

def split_spectra_in_5_sets(spectra: List[Spectrum]) -> List[List[Spectrum]]:
    """Splits a set of spectra into 5 sets with the same number of unique inchikeys"""
    unique_inchikeys = select_unique_inchikeys(spectra)





if __name__ == "__main__":
    # load in cleaned positive mode spectra.
    # Select all unique inchikeys
    # Split in 5, select matching spectra.
    # Train MS2Deepscore and Spec2Vec on 4/5th training spectra
    # Train MS2Query with MS2Deepscore and Spec2Vec
    # Use test set on MS2Query to test performance
    # Store the complete model and the 5 data splits.