from typing import List
from matchms import Spectrum
import numpy as np
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix, jaccard_index
from rdkit import Chem


def get_fingerprint(smiles: str):
    fingerprint = np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles), fpSize=2048))
    assert isinstance(fingerprint, np.ndarray) and fingerprint.sum() > 0, \
        f"Fingerprint for 1 spectrum could not be set smiles is {fingerprint}"
    return fingerprint


def calculate_tanimoto_scores(list_of_smiles_1: List[str],
                              list_of_smiles_2: List[str]) -> np.ndarray:
    """Returns a 2d ndarray containing the tanimoto scores between the smiles"""
    fingerprints_1 = np.array([get_fingerprint(spectrum) for spectrum in list_of_smiles_1])
    fingerprints_2 = np.array([get_fingerprint(spectrum) for spectrum in list_of_smiles_2])
    tanimoto_scores = jaccard_similarity_matrix(fingerprints_1, fingerprints_2)
    return tanimoto_scores


def calculate_single_tanimoto_score(smiles_1: str,
                                    smiles_2: str) -> float:
    """Returns the tanimoto score and a boolean showing if the spectra are exact matches"""
    test_spectrum_fingerprint = get_fingerprint(smiles_1)
    library_fingerprint = get_fingerprint(smiles_2)
    tanimoto_score = float(jaccard_index(library_fingerprint, test_spectrum_fingerprint))
    return tanimoto_score
