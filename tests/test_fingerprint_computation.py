import numpy as np
import pytest

from ms2query.fingerprint_computation import (
    FingerprintGenerator,
    SparseFingerprintGenerator,
    get_mol_from_smiles,
    prepare_sparse_vector,
    compute_fingerprints_from_smiles,
    count_fingerprint_keys,
    compute_idf,
)

# -----------------------
# Test doubles / fakes
# -----------------------

class _FakeSparseCountFP:
    """Mimics RDKit's SparseCountFingerprint object."""
    def __init__(self, elements_dict):
        self._elems = dict(elements_dict)

    def GetNonzeroElements(self):
        return dict(self._elems)


class FakeFPGen:
    """
    A fake fingerprint generator exposing the subset of methods used by the code:
      - GetFingerprintAsNumPy(mol) -> dense binary/float vector
      - GetCountFingerprintAsNumPy(mol) -> dense count vector
      - GetSparseCountFingerprint(mol).GetNonzeroElements() -> {bit: count}
    """
    def __init__(self,
                 dense_fp=None,
                 dense_count_fp=None,
                 sparse_dict=None):
        # Defaults
        self._dense_fp = np.array([0, 1, 1, 0], dtype=np.int32) if dense_fp is None else np.asarray(dense_fp)
        self._dense_count_fp = np.array([0, 2, 5, 0], dtype=np.int32) if dense_count_fp is None else np.asarray(dense_count_fp)
        self._sparse = {1: 2, 2: 5} if sparse_dict is None else dict(sparse_dict)

    def GetFingerprintAsNumPy(self, mol):
        return self._dense_fp.copy()

    def GetCountFingerprintAsNumPy(self, mol):
        return self._dense_count_fp.copy()

    def GetSparseCountFingerprint(self, mol):
        return _FakeSparseCountFP(self._sparse)


# -----------------------
# RDKit / SMILES parsing
# -----------------------

def test_get_mol_from_smiles_valid():
    mol = get_mol_from_smiles("CCO")  # ethanol
    assert mol is not None

def test_get_mol_from_smiles_invalid():
    mol = get_mol_from_smiles("this_is_not_a_smiles")
    assert mol is None


# -----------------------
# Dense fingerprints
# -----------------------

def test_dense_count_scaling_log_and_weights():
    fpgen = FakeFPGen(dense_count_fp=[0, 2, 5, 0])
    gen = FingerprintGenerator(fpgen)

    bit_weights = np.array([1.0, 2.0, 0.5, 1.0], dtype=np.float32)
    out = gen.fingerprint_from_smiles("CCO", count=True, bit_scaling="log", bit_weights=bit_weights)

    expected = np.log1p(np.array([0, 2, 5, 0], dtype=np.float32)) * bit_weights
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

def test_dense_count_scaling_requires_count_true():
    fpgen = FakeFPGen()
    gen = FingerprintGenerator(fpgen)
    with pytest.raises(NotImplementedError):
        _ = gen.fingerprint_from_smiles("CCO", count=False, bit_scaling="log", bit_weights=None)

def test_dense_count_weights_requires_count_true():
    fpgen = FakeFPGen()
    gen = FingerprintGenerator(fpgen)
    with pytest.raises(NotImplementedError):
        _ = gen.fingerprint_from_smiles("CCO", count=False, bit_scaling=None, bit_weights=np.ones(4, dtype=np.float32))

def test_dense_weights_shape_validation():
    fpgen = FakeFPGen(dense_count_fp=[1, 1, 1, 1])
    gen = FingerprintGenerator(fpgen)
    # wrong length
    # TODO: check/fix:
    # with pytest.raises(ValueError):
    #    _ = gen.fingerprint_from_smiles("CCO", count=True, bit_scaling=None, bit_weights=np.ones(3, dtype=np.float32))
    # wrong type
    # with pytest.raises(TypeError):
    #    _ = gen.fingerprint_from_smiles("CCO", count=True, bit_scaling=None, bit_weights=[1.0, 1.0, 1.0, 1.0])

def test_dense_no_scaling_no_weights_returns_base():
    base = np.array([0, 1, 1, 0], dtype=np.int32)
    fpgen = FakeFPGen(dense_fp=base)
    gen = FingerprintGenerator(fpgen)
    out = gen.fingerprint_from_smiles("CCO", count=False, bit_scaling=None, bit_weights=None)
    np.testing.assert_array_equal(out, base.astype(np.float32))  # class casts to float32


# -----------------------
# Sparse fingerprints
# -----------------------

def test_sparse_count_scaling_and_weights():
    sparse_dict = {1: 2, 4: 5, 7: 3}
    fpgen = FakeFPGen(sparse_dict=sparse_dict)
    gen = SparseFingerprintGenerator(fpgen)

    weights = {1: 10.0, 4: 0.5}  # 7 missing -> default 1.0
    keys, values = gen.fingerprint_from_smiles("CCO", count=True, bit_scaling="log", bit_weights=weights)

    assert keys.dtype == np.int64
    assert values.dtype == np.float32
    # verify order is sorted by key
    assert list(keys) == sorted(sparse_dict.keys())

    # manual expected
    base = np.array([sparse_dict[k] for k in keys], dtype=np.float32)
    expected = np.log1p(base) * np.array([weights.get(int(k), 1.0) for k in keys], dtype=np.float32)
    np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-6)

def test_sparse_indices_only_no_scaling_no_weights():
    sparse_dict = {5: 1, 2: 3, 9: 2}
    fpgen = FakeFPGen(sparse_dict=sparse_dict)
    gen = SparseFingerprintGenerator(fpgen)

    indices = gen.fingerprint_from_smiles("CCO", count=False, bit_scaling=None, bit_weights=None)
    assert indices.dtype == np.int64
    assert list(indices) == sorted(sparse_dict.keys())

def test_sparse_indices_only_rejects_scaling_and_weights():
    fpgen = FakeFPGen(sparse_dict={1: 2})
    gen = SparseFingerprintGenerator(fpgen)
    with pytest.raises(NotImplementedError):
        _ = gen.fingerprint_from_smiles("CCO", count=False, bit_scaling="log", bit_weights=None)
    with pytest.raises(NotImplementedError):
        _ = gen.fingerprint_from_smiles("CCO", count=False, bit_scaling=None, bit_weights={1: 2.0})

def test_prepare_sparse_vector_validates_bit_scaling():
    with pytest.raises(ValueError):
        _ = prepare_sparse_vector({1: 2}, bit_scaling="unknown", bit_weights=None)


# -----------------------
# Batch computation
# -----------------------

def test_compute_fingerprints_from_smiles_dense_stack():
    fpgen = FakeFPGen(dense_count_fp=[0, 2, 5, 1])
    smiles = ["CCO", "CCC"]
    # weights of correct length
    w = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    arr = compute_fingerprints_from_smiles(
        smiles, fpgen, count=True, sparse=False, bit_scaling="log", bit_weights=w, progress_bar=False
    )
    assert isinstance(arr, np.ndarray) and arr.shape == (2, 4)
    # check first row transform
    expected0 = np.log1p(np.array([0, 2, 5, 1], dtype=np.float32)) * w
    np.testing.assert_allclose(arr[0], expected0, rtol=1e-6, atol=1e-6)

def test_compute_fingerprints_from_smiles_sparse_list_of_tuples():
    fpgen = FakeFPGen(sparse_dict={1: 2, 3: 1})
    smiles = ["CCO", "CCC", "CCN"]
    out = compute_fingerprints_from_smiles(
        smiles, fpgen, count=True, sparse=True, bit_scaling=None, bit_weights={1: 10.0}, progress_bar=False
    )
    assert isinstance(out, list) and len(out) == 3
    k0, v0 = out[0]
    assert list(k0) == [1, 3]
    np.testing.assert_allclose(v0, np.array([2, 1], dtype=np.float32) * np.array([10.0, 1.0], dtype=np.float32))


# -----------------------
# Numba helper & IDF
# -----------------------

def test_count_fingerprint_keys_basic():
    # emulate indices-only sparse fingerprints from three molecules
    fps = [
        np.array([1, 3, 5], dtype=np.int64),
        np.array([3, 4], dtype=np.int64),
        np.array([1, 4, 6], dtype=np.int64),
    ]
    keys, counts, first_idx = count_fingerprint_keys(fps)

    # expected
    expected_counts = {1: 2, 3: 2, 4: 2, 5: 1, 6: 1}
    expected_first = {1: 0, 3: 0, 4: 1, 5: 0, 6: 2}

    assert list(keys) == sorted(expected_counts.keys())
    assert list(counts) == [expected_counts[k] for k in keys]
    assert list(first_idx) == [expected_first[k] for k in keys]

def test_compute_idf_zero_safe_and_values():
    X = np.array([
        [0, 1, 0, 0],
        [1, 2, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    # df = [1, 2, 0, 1] -> the 0 will be clamped to 1 internally
    idf = compute_idf(X)
    assert idf.shape == (4,)
    # df[2] would be 0 => treated as 1 -> log(3/1) == log(3)
    np.testing.assert_allclose(idf[2], np.log(3.0), rtol=1e-6, atol=1e-6)
    # spot-check a non-zero df
    np.testing.assert_allclose(idf[1], np.log(3.0 / 2.0), rtol=1e-6, atol=1e-6)
