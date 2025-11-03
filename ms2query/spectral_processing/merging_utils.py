import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from matchms import Spectrum


METADATA_FIELDS_FROM_FIRST = [
    "ionmode", "smiles", "inchikey", "inchi", "name", "precursor_mz",
]
METADATA_FIELDS_SUM = ["instrument_type", "adduct", "collision_energy"]


# --------------------- Helper functions ---------------------
def _normalize_spectrum_sum(s):
    """Return a *copy* of s with intensities normalized to sum=1 (if possible)."""
    mz = np.asarray(s.peaks.mz, dtype=float)
    intens = np.asarray(s.peaks.intensities, dtype=float)
    tot = intens.sum()
    if tot > 0:
        intens = intens / tot
    # Build a shallow copy with normalized peaks but same metadata
    md = dict(s.metadata) if hasattr(s, "metadata") else {}
    return Spectrum(mz=mz, intensities=intens, metadata=md)


def _merge_cluster_to_consensus(cluster_spectra, mz_tol=0.01, min_frac=0.25):
    """
    Build a consensus spectrum from a list of matchms Spectra.

    Parameters
    ----------
    mz_tol: float
        Tolerance in Da (set ppm handling yourself if needed).
    min_frac: float
        The minimum fraction of spectra in which a peak must appear to be kept.
    """
    n = len(cluster_spectra)
    if n == 1:
        # Shouldn’t happen here, but keep it safe.
        return _normalize_spectrum_sum(cluster_spectra[0])

    # Normalize each spectrum (sum=1)
    specs = [_normalize_spectrum_sum(s) for s in cluster_spectra]

    # Collect all peaks with a spectrum index
    all_mz = []
    all_int = []
    all_sid = []
    for sid, s in enumerate(specs):
        mz = np.asarray(s.peaks.mz, dtype=float)
        intens = np.asarray(s.peaks.intensities, dtype=float)
        all_mz.append(mz)
        all_int.append(intens)
        all_sid.append(np.full_like(intens, sid, dtype=int))
    all_mz = np.concatenate(all_mz) if len(all_mz) else np.array([], dtype=float)
    all_int = np.concatenate(all_int) if len(all_int) else np.array([], dtype=float)
    all_sid = np.concatenate(all_sid) if len(all_sid) else np.array([], dtype=int)

    if all_mz.size == 0:
        # Edge case: empty cluster
        return Spectrum(mz=np.array([], dtype=float),
                        intensities=np.array([], dtype=float),
                        metadata={"num_merged": n})

    order = np.argsort(all_mz)
    mz_sorted = all_mz[order]
    int_sorted = all_int[order]
    sid_sorted = all_sid[order]

    # Greedy binning
    consensus_mz = []
    consensus_int = []
    i = 0
    while i < mz_sorted.size:
        # start new bin
        bin_mz = [mz_sorted[i]]
        bin_int = [int_sorted[i]]
        bin_sid = {int(sid_sorted[i])}
        # Use a running "center" as intensity-weighted mean to make binning stable
        w_sum = int_sorted[i]
        mz_center = mz_sorted[i]
        j = i + 1
        while j < mz_sorted.size:
            # If next peak is within tolerance to current center, add it
            if abs(mz_sorted[j] - mz_center) <= mz_tol:
                bin_mz.append(mz_sorted[j])
                bin_int.append(int_sorted[j])
                bin_sid.add(int(sid_sorted[j]))
                # update center (weighted)
                w_sum += int_sorted[j]
                mz_center = (mz_center*(w_sum - int_sorted[j]) + mz_sorted[j]*int_sorted[j]) / w_sum
                j += 1
            else:
                break

        # Decide if this bin is frequent enough across spectra
        frac = len(bin_sid) / n
        if frac >= min_frac:
            bin_mz = np.asarray(bin_mz, dtype=float)
            bin_int = np.asarray(bin_int, dtype=float)
            # intensity-weighted centroid m/z
            cz_mz = np.average(bin_mz, weights=bin_int)
            # sum intensities conveys both frequency and strength
            cz_int = bin_int.sum()
            consensus_mz.append(cz_mz)
            consensus_int.append(cz_int)

        # move to next bin
        i = j

    if len(consensus_mz) == 0:
        # If nothing survives the min_frac filter, relax by keeping top-K overall peaks
        # (K=50 here as a fallback)
        K = 50
        # Recompute simple top-K by summed intensities without the min_frac gate
        # (this block runs rarel, just it's a safety net)
        # Group again but keep everything:
        # TODO: clean this up...
        consensus_mz = []
        consensus_int = []
        i = 0
        while i < mz_sorted.size:
            bin_mz = [mz_sorted[i]]
            bin_int = [int_sorted[i]]
            w_sum = int_sorted[i]
            mz_center = mz_sorted[i]
            j = i + 1
            while j < mz_sorted.size and abs(mz_sorted[j] - mz_center) <= mz_tol:
                bin_mz.append(mz_sorted[j])
                bin_int.append(int_sorted[j])
                w_sum += int_sorted[j]
                mz_center = (mz_center*(w_sum - int_sorted[j]) + mz_sorted[j]*int_sorted[j]) / w_sum
                j += 1
            bin_mz = np.asarray(bin_mz, dtype=float)
            bin_int = np.asarray(bin_int, dtype=float)
            consensus_mz.append(np.average(bin_mz, weights=bin_int))
            consensus_int.append(bin_int.sum())
            i = j
        # keep top-K by intensity
        if len(consensus_int) > K:
            idx = np.argsort(consensus_int)[-K:]
            consensus_mz = list(np.asarray(consensus_mz)[idx])
            consensus_int = list(np.asarray(consensus_int)[idx])

    consensus_mz = np.asarray(consensus_mz, dtype=float)
    consensus_int = np.asarray(consensus_int, dtype=float)

    # Final renormalization (sum=1)
    ssum = consensus_int.sum()
    if ssum > 0:
        consensus_int = consensus_int / ssum

    # Metadata: inherit metadata from first spectrum:
    md = {}
    for field in METADATA_FIELDS_FROM_FIRST:
        if hasattr(cluster_spectra[0], "metadata") and field in cluster_spectra[0].metadata:
            md[field] = cluster_spectra[0].metadata[field]

    # add all entries for METADATA_FIELDS_SUM by string concatenation:
    for field in METADATA_FIELDS_SUM:
        vals = []
        for s in cluster_spectra:
            if hasattr(s, "metadata") and field in s.metadata and s.metadata[field] is not None:
                vals.append(str(s.metadata[field]))
        if vals:
            md[field] = ";".join(vals)
    md["num_merged"] = n

    return Spectrum(mz=consensus_mz, intensities=consensus_int, metadata=md)


# --------------------- Main functions ---------------------
def get_merged_spectra(spectra, clusters, mz_tol=0.01, min_frac=0.25):
    """Given a list of spectra and clusters (lists of indices into spectra),
    return a list of merged consensus spectra.

    Parameters
    ----------
    spectra: list[matchms.Spectrum]
        List of matchms Spectrum objects to be merged.
    clusters: 
        List of lists/arrays with indices into `spectra`. Each sublist defines a cluster to be merged.
    mz_tol: float
        m/z tolerance in Da for merging peaks.
    min_frac: float
        Keep a consensus peak only if present in >= this fraction of spectra.
    """
    spectra_new = []
    for cluster in clusters:
        if len(cluster) > 1:
            cluster_spectra = [spectra[i] for i in cluster]
            merged = _merge_cluster_to_consensus(cluster_spectra, mz_tol=mz_tol, min_frac=min_frac)
            spectra_new.append(merged)
        else:
            # singletons: normalize to sum=1 for consistency
            spectra_new.append(_normalize_spectrum_sum(spectra[cluster[0]]))
    return spectra_new


def cluster_block(spectra, sim_score, threshold=0.95):
    """Find clusters of highly similar spectra (according to Cosine).
    Hint: Use lower intensity_power to emphasize smaller peaks.

    Parameters
    ----------
    spectra:
        List of matchms Spectrum objects.
    sim_score:
        Matchms scoring method, e.g. CosineGreedy()
    threshold: float
        Spectra with similarity >= threshold will be merged.
    """
    # similarity
    sim = sim_score.matrix(spectra, spectra, is_symmetric=True)
    S = sim["score"]

    # Graph by threshold on upper triangle
    iu = np.triu_indices_from(S, 1)
    edges = np.where(S[iu] >= threshold)[0]
    rows = iu[0][edges]
    cols = iu[1][edges]
    n = S.shape[0]
    G = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n, n))
    G = G + G.T + csr_matrix(np.eye(n))

    # Connected components = clusters
    n_comp, labels = connected_components(G, directed=False)
    clusters = [np.where(labels == k)[0] for k in range(n_comp)]
    return clusters, S
