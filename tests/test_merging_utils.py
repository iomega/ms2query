# test_merge_spectra.py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from matchms import Spectrum

# Adjust if your module name is different:
from ms2query.spectral_processing.merging_utils import (
    _normalize_spectrum_sum,
    _merge_cluster_to_consensus,
    get_merged_spectra,
)

# ---------- helpers ----------

def spec(mz, intensities, **metadata):
    """Small helper to build a matchms Spectrum."""
    return Spectrum(mz=np.asarray(mz, dtype=float),
                    intensities=np.asarray(intensities, dtype=float),
                    metadata=metadata or {})

# ---------- unit tests ----------

def test_normalize_spectrum_sum():
    s = spec([100, 150], [10, 5])
    sn = _normalize_spectrum_sum(s)
    assert pytest.approx(sn.peaks.intensities.sum()) == 1.0
    # Same mz layout
    assert_allclose(sn.peaks.mz, [100, 150])

def test_merge_two_identical_spectra_keeps_shape_and_sets_metadata():
    # Two identical-peak spectra (different scaling)
    s1 = spec([100.0, 150.0], [10.0, 5.0], name="a")
    s2 = spec([100.0, 150.0], [20.0, 5.0], name="b")

    consensus = _merge_cluster_to_consensus([s1, s2], mz_tol=0.01, min_frac=0.5)

    # Consensus must have both peaks and sum to 1
    assert len(consensus.peaks.mz) == 2
    assert pytest.approx(consensus.peaks.intensities.sum()) == 1.0

    # Ratios of consensus intensities reflect sums of per-spectrum normalized intensities
    # s1 norm: [10/15, 5/15], s2 norm: [20/25, 5/25]
    expected_100 = (10/15) + (20/25)
    expected_150 = (5/15) + (5/25)
    total = expected_100 + expected_150
    exp = np.array([expected_100/total, expected_150/total])

    # Ensure peak order matches m/z order
    order = np.argsort(consensus.peaks.mz)
    assert_allclose(consensus.peaks.mz[order], [100.0, 150.0])
    assert_allclose(consensus.peaks.intensities[order], exp, rtol=1e-6, atol=1e-6)

    # Metadata should record number merged
    assert consensus.metadata.get("num_merged") == 2

def test_min_frac_filters_rare_peak():
    # 4 spectra; peak at 200 appears once -> should be dropped if min_frac=0.5
    s1 = spec([100, 200], [1, 1])
    s2 = spec([100], [1])
    s3 = spec([100], [2])
    s4 = spec([100], [3])

    c = _merge_cluster_to_consensus([s1, s2, s3, s4], mz_tol=0.01, min_frac=0.5)
    # Only m/z ~100 survives
    assert len(c.peaks.mz) == 1
    assert pytest.approx(c.peaks.mz[0], rel=0, abs=1e-6) == 100.0
    assert pytest.approx(c.peaks.intensities.sum()) == 1.0

def test_mz_tolerance_bins_close_peaks():
    # Two peaks within tolerance should collapse into one bin
    s1 = spec([100.000], [1.0])
    s2 = spec([100.007], [1.0])  # within 0.01 Da
    c = _merge_cluster_to_consensus([s1, s2], mz_tol=0.01, min_frac=0.5)
    assert len(c.peaks.mz) == 1
    # centroid must be between the two
    assert 100.000 <= c.peaks.mz[0] <= 100.007
    assert pytest.approx(c.peaks.intensities.sum()) == 1.0

def test_fallback_when_no_bin_meets_min_frac():
    # 3 spectra with disjoint peaks; min_frac=0.5 -> nothing passes, triggers fallback
    s1 = spec([100.0], [1.0])
    s2 = spec([150.0], [1.0])
    s3 = spec([200.0], [1.0])

    c = _merge_cluster_to_consensus([s1, s2, s3], mz_tol=0.01, min_frac=0.5)
    # Fallback keeps all three bins and normalizes
    assert len(c.peaks.mz) == 3
    assert pytest.approx(c.peaks.intensities.sum()) == 1.0

def test_empty_peaks_cluster_gives_empty_consensus():
    # Edge case: all spectra are empty
    s1 = spec([], [])
    s2 = spec([], [])
    c = _merge_cluster_to_consensus([s1, s2], mz_tol=0.01, min_frac=0.5)
    assert c.peaks.mz.size == 0
    assert c.peaks.intensities.size == 0
    assert c.metadata.get("num_merged") == 2

def test_get_merged_spectra_mixes_singletons_and_merged():
    # clusters: one singleton, one 2-member cluster that merges
    spectra = [
        spec([100, 150], [1, 1]),   # 0
        spec([200], [1]),           # 1
        spec([200.005], [2]),       # 2 (merges with #1 under 0.01 Da)
    ]
    clusters = [[0], [1, 2]]

    merged = get_merged_spectra(spectra, clusters, mz_tol=0.01, min_frac=0.5)

    # We should get two spectra back: normalized singleton and merged consensus
    assert len(merged) == 2

    # First is the normalized singleton
    s_single = merged[0]
    assert_allclose(s_single.peaks.mz, [100.0, 150.0])
    assert pytest.approx(s_single.peaks.intensities.sum()) == 1.0

    # Second is the consensus around m/z ~200
    s_cons = merged[1]
    assert len(s_cons.peaks.mz) == 1
    assert abs(s_cons.peaks.mz[0] - 200.0) <= 0.01
    assert pytest.approx(s_cons.peaks.intensities.sum()) == 1.0
    assert s_cons.metadata.get("num_merged") == 2
