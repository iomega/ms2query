"""This code is all taken from the "count your bits" paper:

TODO: only keep what is needed. And, add tests.
"""

import numba
from numba import prange
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances


@numba.njit
def jaccard_similarity_matrix_weighted(references: np.ndarray, queries: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Returns matrix of weighted jaccard indices between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].
    weights
        weighting factor as 1D numpy array. Contains non-negative weights that
        define the importance of each feature in the similarity computation.

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = jaccard_index_weighted(references[i, :], queries[j, :], weights)
    return scores


@numba.njit
def jaccard_index(u: np.ndarray, v: np.ndarray) -> np.float64:
    r"""Computes a weighted Jaccard-index (or Jaccard similarity coefficient) of two boolean
    1-D arrays.
    The Jaccard index between 1-D boolean arrays `u` and `v`,
    is defined as

    .. math::

       J(u,v) = \\frac{u \cap v}
                {u \cup v}

    Parameters
    ----------
    u :
        Input array. Expects boolean vector.
    v :
        Input array. Expects boolean vector.

    Returns
    -------
    jaccard_similarity
        The Jaccard similarity coefficient between vectors `u` and `v`.
    """
    u_or_v = np.bitwise_or(u != 0, v != 0)
    u_and_v = np.bitwise_and(u != 0, v != 0)
    jaccard_score = 0
    if u_or_v.sum() != 0:
        u_or_v = u_or_v
        u_and_v = u_and_v
        jaccard_score = np.float64(u_and_v.sum()) / np.float64(u_or_v.sum())
    return jaccard_score


@numba.njit
def jaccard_index_weighted(u: np.ndarray, v: np.ndarray, weights: np.ndarray) -> np.float64:
    r"""Computes a weighted Jaccard-index (or Jaccard similarity coefficient) of two boolean
    1-D arrays.
    The Jaccard index between 1-D boolean arrays `u` and `v`,
    is defined as

    .. math::

       J(u,v) = \\frac{u \cap v}
                {u \cup v}

    Parameters
    ----------
    u :
        Input array. Expects boolean vector.
    v :
        Input array. Expects boolean vector.
    weights :
        1D array of non-negative floats. Specifies the weight for each dimension; must
        be the same length as `u` and `v`.

    Returns
    -------
    jaccard_similarity
        The Jaccard similarity coefficient between vectors `u` and `v`.
    """
    u_or_v = np.bitwise_or(u != 0, v != 0)
    u_and_v = np.bitwise_and(u != 0, v != 0)
    jaccard_score = 0
    if u_or_v.sum() != 0:
        u_or_v = u_or_v * weights
        u_and_v = u_and_v * weights
        jaccard_score = np.float64(u_and_v.sum()) / np.float64(u_or_v.sum())
    return jaccard_score


@numba.njit
def jaccard_index_sparse(keys1, keys2) -> float:
    """
    Calculate the Jaccard similarity between two sparse binary vectors.

    Parameters:
    keys1, keys2 (array-like): Keys for the first and second sparse vectors (sorted arrays).

    Returns:
    float: The Jaccard similarity (or index).
    """
    i, j = 0, 0
    intersection = 0

    # Traverse both key arrays
    while i < len(keys1) and j < len(keys2):
        if keys1[i] == keys2[j]:
            intersection += 1
            i += 1
            j += 1
        elif keys1[i] < keys2[j]:
            i += 1
        else:
            j += 1

    # Calculate union size
    union = len(keys1) + len(keys2) - intersection

    return intersection / union if union > 0 else 0.0


def jaccard_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of jaccard indices between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    # The trick to fast inference is to use float32 since it allows using BLAS
    references = np.array(references, dtype=np.float32)  # R,N
    queries = np.array(queries, dtype=np.float32)  # Q,N
    intersection = references @ queries.T  # R,N @ N,Q -> R,Q
    union = np.sum(references, axis=1, keepdims=True) + np.sum(queries,axis=1, keepdims=True).T  # R,1+1,Q -> R,Q
    union -= intersection
    jaccard = np.nan_to_num(intersection / union)  # R,Q
    return jaccard


@numba.jit(nopython=True, fastmath=True, parallel=True)
def jaccard_similarity_matrix_sparse(
        references: list, queries: list) -> np.ndarray:
    """Returns matrix of Jaccard similarity between all-vs-all vectors of references and queries.

    Parameters
    ----------
    references:
        List of sparse fingerprints (arrays with keys).
    queries
        List of sparse fingerprints (arrays with keys).

    Returns
    -------
    scores:
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = len(references)
    size2 = len(queries)
    scores = np.zeros((size1, size2))
    for i in prange(size1):
        for j in range(size2):
            scores[i, j] = jaccard_index_sparse(
                references[i],
                queries[j])
    return scores


@numba.njit
def generalized_tanimoto_similarity(A, B):
    """
    Calculate the generalized Tanimoto similarity between two count vectors.
    
    Parameters:
    A (array-like): First count vector.
    B (array-like): Second count vector.
    
    Returns:
    float: Tanimoto similarity.
    """
    
    min_sum = np.sum(np.minimum(A, B))
    max_sum = np.sum(np.maximum(A, B))
    
    return min_sum / max_sum


def generalized_tanimoto_similarity_matrix_sparse_all_vs_all(fingerprints) -> np.ndarray:
    """
    Calculate the generalized Tanimoto similarity between all sparse fingerprints.
    """
    mapping = occupied_bit_mapping(fingerprints)
    X = sparse_fingerprint_to_csr(fingerprints, mapping)

    # Precompute L1 norms.
    norms = np.array(X.sum(axis=1)).ravel()

    # Compute pairwise Manhattan distances.
    manhattan = pairwise_distances(X, metric='manhattan')

    return compute_generalized_tanimoto_from_manhattan_symmetric(norms, manhattan)


def generalized_tanimoto_similarity_matrix_sparse(fingerprints_1, fingerprints_2) -> np.ndarray:
    """
    Calculate the generalized Tanimoto similarity between sparse fingerprints_1 and fingerprints_2.
    """
    mapping = occupied_bit_mapping(fingerprints_1 + fingerprints_2)
    X1 = sparse_fingerprint_to_csr(fingerprints_1, mapping)
    X2 = sparse_fingerprint_to_csr(fingerprints_2, mapping)

    # Precompute L1 norms.
    norms1 = np.array(X1.sum(axis=1)).ravel()
    norms2 = np.array(X2.sum(axis=1)).ravel()

    # Compute pairwise Manhattan distances.
    manhattan = pairwise_distances(X1, X2, metric='manhattan')

    return compute_generalized_tanimoto_from_manhattan(norms1, norms2, manhattan)


def occupied_bit_mapping(fingerprints_sparse):
    # Collect all unique keys.
    all_keys = set()
    for keys, _ in fingerprints_sparse:
        all_keys.update(keys.tolist())

    # Create a mapping from original key to a new, contiguous index.
    sorted_keys = sorted(all_keys)
    return {old_key: new_key for new_key, old_key in enumerate(sorted_keys)}


def sparse_fingerprint_to_csr(fingerprints_sparse, mapping):
    # Build lists for constructing the sparse matrix.
    rows = []
    cols = []
    vals = []

    for i, (keys, values) in enumerate(fingerprints_sparse):
        new_keys = np.array([mapping[k] for k in keys], dtype=np.int32)
        rows.extend([i] * len(new_keys))
        cols.extend(new_keys.tolist())
        vals.extend(values.tolist())

    num_rows = len(fingerprints_sparse)
    num_cols = len(mapping)  # number of occupied features

    # Build the COO matrix and convert to CSR.
    X = sp.coo_matrix((vals, (rows, cols)), shape=(num_rows, num_cols), dtype=np.float32)
    return X.tocsr()


@numba.njit(parallel=True, fastmath=True)
def compute_generalized_tanimoto_from_manhattan_symmetric(norms, manhattan):
    n = norms.shape[0]
    tanimoto = np.empty((n, n), dtype=norms.dtype)
    for i in numba.prange(n):
        for j in range(n):
            union = norms[i] + norms[j] + manhattan[i, j]
            if union > 0:
                tanimoto[i, j] = (norms[i] + norms[j] - manhattan[i, j]) / union
            else:
                tanimoto[i, j] = 1.0
    return tanimoto


@numba.njit(parallel=True, fastmath=True)
def compute_generalized_tanimoto_from_manhattan(norms1, norms2, manhattan):
    n = norms1.shape[0]
    m = norms2.shape[0]
    tanimoto = np.empty((n, m), dtype=norms1.dtype)
    for i in numba.prange(n):
        for j in range(m):
            union = norms1[i] + norms2[j] + manhattan[i, j]
            if union > 0:
                tanimoto[i, j] = (norms1[i] + norms2[j] - manhattan[i, j]) / union
            else:
                tanimoto[i, j] = 1.0
    return tanimoto


@numba.njit
def generalized_tanimoto_similarity_sparse_numba(keys1, values1, keys2, values2) -> float:
    """
    Calculate the generalized Tanimoto similarity between two sparse count vectors.

    Parameters:
    keys1, values1 (array-like): Keys and values for the first sparse vector.
    keys2, values2 (array-like): Keys and values for the second sparse vector.
    """
    i, j = 0, 0
    min_sum, max_sum = 0.0, 0.0

    while i < len(keys1) and j < len(keys2):
        if keys1[i] == keys2[j]:
            min_sum += min(values1[i], values2[j])
            max_sum += max(values1[i], values2[j])
            i += 1
            j += 1
        elif keys1[i] < keys2[j]:
            max_sum += values1[i]
            i += 1
        else:
            max_sum += values2[j]
            j += 1

    # Add remaining values from both vectors
    while i < len(keys1):
        max_sum += values1[i]
        i += 1

    while j < len(keys2):
        max_sum += values2[j]
        j += 1

    return min_sum / max_sum


@numba.jit(nopython=True, fastmath=True, parallel=True)
def generalized_tanimoto_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of generalized Tanimoto similarity between all-vs-all vectors
    of references and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    assert references.shape[1] == queries.shape[1], "Vector sizes do not match!"

    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2)) #, dtype=np.float32)
    for i in prange(size1):
        for j in range(size2):
            scores[i, j] = generalized_tanimoto_similarity(references[i, :], queries[j, :])
    return scores


@numba.jit(nopython=True, fastmath=True, parallel=True)
def generalized_tanimoto_similarity_matrix_sparse_numba(
    references: list, queries: list) -> np.ndarray:
    """Returns matrix of generalized Tanimoto similarity between all-vs-all vectors of references and queries.

    Parameters
    ----------
    references:
        List of sparse fingerprints (tuple of two arrays: keys and counts).
    queries
        List of sparse fingerprints (tuple of two arrays: keys and counts).

    Returns
    -------
    scores:
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = len(references)
    size2 = len(queries)
    scores = np.zeros((size1, size2))
    for i in prange(size1):
        for j in range(size2):
            scores[i, j] = generalized_tanimoto_similarity_sparse_numba(
                references[i][0], references[i][1],
                queries[j][0], queries[j][1])
    return scores


@numba.njit
def generalized_tanimoto_similarity_weighted(A, B, weights):
    """
    Calculate the weighted generarlized Tanimoto similarity between two count vectors.
    
    Parameters:
    ----------
        A (array-like): First count vector.
        B (array-like): Second count vector.
        weights: weights for every vector bit
    
    Returns:
    float: Tanimoto similarity.
    """
    
    min_sum = np.sum(np.minimum(A, B) * weights)
    max_sum = np.sum(np.maximum(A, B) * weights)
    
    return min_sum / max_sum


@numba.jit(nopython=True, fastmath=True, parallel=True)
def generalized_tanimoto_similarity_matrix_weighted(references: np.ndarray, queries: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Returns matrix of generalized Tanimoto similarity between all-vs-all vectors of references and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2)) #, dtype=np.float32)
    for i in prange(size1):
        for j in range(size2):
            scores[i, j] = generalized_tanimoto_similarity_weighted(references[i, :], queries[j, :], weights)
    return scores


def compute_cosine_greedy(cosine_obj, spectra):
    # This is only a replacement of the matchme method until that will allow disabling tqdm
    n_rows = n_cols = len(spectra)

    idx_row = []
    idx_col = []
    scores = []
    # Wrap the outer loop with tqdm to track progress
    for i_ref, reference in enumerate(spectra[:n_rows]):
        for i_query, query in enumerate(spectra[i_ref:n_cols], start=i_ref):
            score = cosine_obj.pair(reference, query)
            if cosine_obj.keep_score(score):
                idx_row += [i_ref, i_query]
                idx_col += [i_query, i_ref]
                scores += [score, score]

    idx_row = np.array(idx_row, dtype=np.int_)
    idx_col = np.array(idx_col, dtype=np.int_)
    scores_data = np.array(scores, dtype=cosine_obj.score_datatype)
    # TODO: make StackedSparseArray the default and add fixed function to output different formats (with code below)

    scores_array = np.zeros(shape=(n_rows, n_cols), dtype=cosine_obj.score_datatype)
    scores_array[idx_row, idx_col] = scores_data.reshape(-1)
    return scores_array
