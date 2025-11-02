import numpy as np
import numba
from numba import types, typed
from rdkit import Chem
from tqdm import tqdm


class FingerprintGenerator:
    """
    Dense (NumPy) fingerprints.

    Supports:
      - count=True: count dense vector (supports scaling and weights)
      - count=False: binary/float dense vector (no scaling/weights)
    """
    def __init__(self, fpgen):
        self.fpgen = fpgen

    def fingerprint_from_smiles(self, smiles, count=False, bit_scaling=None, bit_weights=None):
        """Compute fingerprint from SMILES using the generator attribute.
        
        Parameters
        ----------
        smiles : str
        count : bool
            If True, returns count fingerprint. Else standard fingerprint.
        bit_scaling : None or 'log'
            Count-scaling: log(1+count) if 'log'. Only valid when count=True.
        bit_weights : np.ndarray or None
            For dense count fingerprints (count=True), provide a 1D float array of the same
            length as the fingerprint. Values are multiplied elementwise.

        Returns
        -------
        np.ndarray or None
        """
        if (bit_scaling is not None) and not count:
            raise NotImplementedError("Scaling is only implemented for dense count fingerprints (count=True).")
        if (bit_weights is not None) and not count:
            raise NotImplementedError("Weights are only implemented for dense count fingerprints (count=True).")

        mol = get_mol_from_smiles(smiles)
        if mol is None:
            return None

        try:
            fp = self.fpgen.GetCountFingerprintAsNumPy(mol) if count else self.fpgen.GetFingerprintAsNumPy(mol)
            fp = fp.astype(np.float32, copy=False)

            # Apply scaling (count vectors only)
            if bit_scaling is not None:
                if bit_scaling.lower() != "log":
                    raise ValueError("bit_scaling must be None or 'log'.")
                fp = np.log1p(fp).astype(np.float32, copy=False)

            # Apply weights (count vectors only)
            if bit_weights is not None:
                if not isinstance(bit_weights, np.ndarray):
                    raise TypeError("bit_weights must be a NumPy 1D float array for dense fingerprints.")
                if bit_weights.ndim != 1 or bit_weights.shape[0] != fp.shape[0]:
                    raise ValueError(f"bit_weights must have shape ({fp.shape[0]},), got {bit_weights.shape}.")
                fp = (fp * bit_weights.astype(np.float32, copy=False)).astype(np.float32, copy=False)

            return fp

        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None


class SparseFingerprintGenerator:
    """
    Sparse fingerprints.

    - count=True: returns (keys, values) where keys are sorted bit indices (int64)
      and values are (optionally scaled/weighted) counts (float32).
    - count=False: returns sorted bit indices as int64 array (no scaling/weights).
    """
    def __init__(self, fpgen):
        self.fpgen = fpgen

    def fingerprint_from_smiles(
        self, smiles: str,
        count: bool = False,
        bit_scaling: str = None,
        bit_weights: dict = None
    ):
        """Compute sparse fingerprint from SMILES using the generator attribute.
        
        Parameters
        ----------
        smiles : str
        count : bool
            If True, returns sparse count fingerprint; else indices only.
        bit_scaling : None or 'log'
            Applies to count=True only. Uses log1p.
        bit_weights : dict[int, float] or None
            Applies to count=True only. Missing bits default to 1.0.

        Returns
        -------
        If count=True: (keys: np.uint32 array, values: np.float32 array)
        If count=False: keys-only np.uint32 array
        """
        if (bit_scaling is not None) and not count:
            raise NotImplementedError("Scaling is only implemented for sparse count fingerprints (count=True).")
        if (bit_weights is not None) and not count:
            raise NotImplementedError("Weights are only implemented for sparse count fingerprints (count=True).")
        
        mol = get_mol_from_smiles(smiles)
        if mol is None:
            return None

        try:
            if count:
                fp_dict = self.fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
                return prepare_sparse_vector(fp_dict, bit_scaling, bit_weights)
            else:
                # Indices only
                fp_elements = self.fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
                return np.array(sorted(fp_elements.keys()), dtype=np.uint32)
        except Exception as e:
            print(f"Error generating fingerprint for SMILES {smiles}: {e}")
            return None


def get_mol_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("MolFromSmiles returned None with default sanitization.")
    except Exception as e:
        print(f"Error processing SMILES {smiles} with default sanitization: {e}")
        print("Retrying with sanitize=False...")
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            # Regenerate computed properties like implicit valence and ring information
            mol.UpdatePropertyCache(strict=False)

            # Apply several sanitization rules (taken from http://rdkit.org/docs/Cookbook.html)
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                catchErrors=True
            )
            if mol is None:
                raise ValueError("MolFromSmiles returned None even with sanitize=False.")
        except Exception as e2:
            print(f"Error processing SMILES {smiles} with sanitize=False: {e2}")
            return None
    return mol


def prepare_sparse_vector(
    sparse_fp_dict: dict,
    bit_scaling: str = None,
    bit_weights: dict = None
):
    """Convert sparse count dict to sorted arrays with optional scaling/weighting.

    Returns
    -------
    keys:   uint32 array of sorted bit indices
    values: float32 array of scaled/weighted counts
    """
    keys = np.array(sorted(sparse_fp_dict.keys()), dtype=np.uint32)
    counts = np.array([float(sparse_fp_dict[k]) for k in keys], dtype=np.float32)

    # scaling
    if bit_scaling is not None:
        if bit_scaling.lower() != "log":
            raise ValueError("bit_scaling must be None or 'log'.")
        counts = np.log1p(counts).astype(np.float32, copy=False)

    # weights
    if bit_weights is not None:
        if not isinstance(bit_weights, dict):
            raise TypeError("For sparse fingerprints, bit_weights must be a dict {bit: weight}.")
        weights = np.array([float(bit_weights.get(int(k), 1.0)) for k in keys], dtype=np.float32)
        counts = (counts * weights).astype(np.float32, copy=False)

    return keys, counts


def compute_fingerprints_from_smiles(
    smiles_lst,
    fpgen,
    count=True,
    sparse=True,
    bit_scaling=None,
    bit_weights=None,
    progress_bar=False,
):
    fp_generator = SparseFingerprintGenerator(fpgen) if sparse else FingerprintGenerator(fpgen)
    
    fingerprints = []
    for i, smiles in tqdm(enumerate(smiles_lst), total=len(smiles_lst), disable=(not progress_bar)):
        fp = fp_generator.fingerprint_from_smiles(
            smiles, count=count, bit_scaling=bit_scaling, bit_weights=bit_weights
        )
        if fp is None:
            print(f"Missing fingerprint for element {i}: {smiles}")
        else:
            fingerprints.append(fp)
    if sparse:
        return fingerprints
    return np.stack(fingerprints)


@numba.njit
def count_fingerprint_keys(fingerprints):
    """
    Count the occurrences of keys across all sparse (indices-only) fingerprints
    using two dictionaries (one for counts, one for first index) for fast lookup.
    
    Parameters
    ----------
    fingerprints : iterable of 1D arrays of int64 bit indices (from count=False sparse).
    
    Returns
    -------
    unique_keys : int64 array
    counts : int32 array
    first_instances : int32 array
    """
    counts = typed.Dict.empty(key_type=types.int64, value_type=types.int32)
    first_instance = typed.Dict.empty(key_type=types.int64, value_type=types.int32)
    
    for i, fp_bits in enumerate(fingerprints):
        for bit in fp_bits:
            if bit in counts:
                counts[bit] += 1
            else:
                counts[bit] = 1
                first_instance[bit] = i
    
    n = len(counts)
    unique_keys = np.empty(n, dtype=np.int64)
    count_arr   = np.empty(n, dtype=np.int32)
    first_arr   = np.empty(n, dtype=np.int32)
    
    idx = 0
    for key in counts:
        unique_keys[idx] = key
        count_arr[idx] = counts[key]
        first_arr[idx] = first_instance[key]
        idx += 1
    
    order = np.argsort(unique_keys)
    return unique_keys[order], count_arr[order], first_arr[order]


### ------------------------
### Bit Scaling and Weighing
### ------------------------

def compute_idf(vector_array):
    """Compute inverse document frequency (IDF)."""
    N = vector_array.shape[0]
    df = (vector_array > 0).sum(axis=0)
    # avoid divide-by-zero when a column is all zeros
    df = np.where(df == 0, 1, df)
    return np.log(N / df)
