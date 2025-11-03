from typing import List, Dict, Tuple, Optional
import sqlite3
import json
from collections import Counter
import numpy as np
from matchms import Spectrum
from matchms.similarity import CosineGreedy

from ms2query.spectral_processing import cluster_block, get_merged_spectra
from ms2query.spectral_processing.merging_utils import METADATA_FIELDS_FROM_FIRST, METADATA_FIELDS_SUM
from .database_utils import ndarray_to_blob


# ---------------------SQLite: schema & IO  --------------------------
def ensure_merged_tables(conn: sqlite3.Connection) -> None:
    """
    Create tables if missing.
    # TODO: add alter table option as well?
    """
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS merged_spectra (
            merged_id        INTEGER PRIMARY KEY,
            comp_id          TEXT NOT NULL,
            ionmode          TEXT,            -- from first
            charge           INTEGER,
            precursor_mz     REAL NOT NULL,   -- required for MS2DeepScore
            smiles           TEXT,            -- from first
            inchikey         TEXT,            -- from first
            inchi            TEXT,            -- from first
            name             TEXT,            -- from first
            instrument_type  TEXT,            -- majority vote
            adduct           TEXT,            -- majority vote
            collision_energy TEXT,            -- majority vote (e.g., "NCE 20")
            num_merged       INTEGER NOT NULL,
            source_spec_ids  TEXT NOT NULL,   -- JSON list
            mz               BLOB NOT NULL,
            intensities      BLOB NOT NULL,

            -- optional histograms for audit/debug (JSON: {value: count, ...})
            instrument_type_hist  TEXT,
            adduct_hist           TEXT,
            collision_energy_hist TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS merged_embeddings (
            merged_id   INTEGER PRIMARY KEY,
            embedding   BLOB NOT NULL,
            FOREIGN KEY (merged_id) REFERENCES merged_spectra(merged_id) ON DELETE CASCADE
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_merged_spectra_comp ON merged_spectra(comp_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_merged_spectra_mode_charge ON merged_spectra(ionmode, charge);")
    conn.commit()

def _clean_val(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def _mode_summary(values: list[str | None]) -> tuple[Optional[str], dict]:
    """
    Return (mode_value, histogram_dict) ignoring None/empty.
    If all are None, returns (None, {}).
    """
    vals = [_clean_val(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, {}
    c = Counter(vals)
    mode_val, _ = max(c.items(), key=lambda kv: (kv[1], kv[0]))
    return mode_val, dict(c)

def _aggregate_metadata_for_cluster(cluster_specs: list[Spectrum]) -> dict:
    """
    Build a metadata dict for a merged spectrum:
    - fields from first:  METADATA_FIELDS_FROM_FIRST
    - fields by majority: METADATA_FIELDS_SUM (+ *_hist JSON for audit)
    - precursor_mz: median of available precursor_mz across the cluster (float)
    """
    assert len(cluster_specs) >= 1
    first = cluster_specs[0]
    md: dict = {}

    # from first
    for k in METADATA_FIELDS_FROM_FIRST:
        md[k] = _safe_meta(first, k, None)

    # majority vote fields (+ histograms)
    for k in METADATA_FIELDS_SUM:
        vals = [_safe_meta(s, k, None) for s in cluster_specs]
        mode_val, hist = _mode_summary(vals)
        md[k] = mode_val
        md[f"{k}_hist"] = json.dumps(hist) if hist else None

    # precursor_mz: use median over cluster (fallback to first if none)
    precs = []
    for s in cluster_specs:
        v = _safe_meta(s, "precursor_mz", None)
        try:
            if v is not None:
                precs.append(float(v))
        except Exception:
            pass
    if precs:
        md["precursor_mz"] = float(np.median(precs))
    else:
        # last-resort fallback (may still be None)
        md["precursor_mz"] = _safe_meta(first, "precursor_mz", None)

    return md


# ------------------------ helper functions --------------------------

def _safe_meta(s: Spectrum, key: str, default=None):
    try:
        return s.metadata.get(key, default) if hasattr(s, "metadata") else default
    except Exception:
        return default


def _split_by_mode_charge(spectra: List[Spectrum]) -> Dict[Tuple[str, Optional[int]], List[int]]:
    """
    Group indices by (ionmode, charge).
    ionmode is lowercased; missing -> 'unknown'; charge may be None.
    """
    groups: Dict[Tuple[str, Optional[int]], List[int]] = {}
    for i, s in enumerate(spectra):
        ionmode = _safe_meta(s, "ionmode", "unknown")
        ionmode = str(ionmode).lower() if ionmode is not None else "unknown"
        charge = _safe_meta(s, "charge", None)
        groups.setdefault((ionmode, charge), []).append(i)
    return groups


# --------------------------- cluster & merge ---------------------------

def cluster_and_merge_to_sqlite(
    mapper,
    sdb,
    conn: sqlite3.Connection,
    *,
    cosine_thr: float = 0.95,
    intensity_power: float = 0.5,
    mz_tol: float = 0.01,
    min_frac: float = 0.5,
    commit_every: int = 50,
    skip_if_comp_done: bool = False,
) -> int:
    """
    Cluster & merge spectra per (comp_id, ionmode, charge) and write results to `merged_spectra`.

    Parameters
    ----------
    mapper : Any
        Has `get_all_mappings()` -> DataFrame with columns ['comp_id', 'spec_id'].
    sdb : Any
        Has `get_spectra_by_ids(List[int]) -> List[matchms.Spectrum]`.
    conn : sqlite3.Connection
        Connection to the *same* SQLite DB as `sdb` (so tables live together).
    cosine_thr : float
        Cosine threshold used in `cluster_block` (connected components).
    intensity_power : float
        Intensity power transform for similarity only (I := I**p).
    mz_tol : float
        m/z tolerance (Da) used by `get_merged_spectra` for peak alignment.
    min_frac : float
        Keep consensus peaks present in ≥ this fraction of spectra in cluster.
    commit_every : int
        Commit after processing this many compounds for resilience.
    skip_if_comp_done : bool
        If True, skip a compound when at least one row with its comp_id exists
        in `merged_spectra` (useful for resumable runs).

    Returns
    -------
    int
        Number of merged spectra rows inserted.
    """
    ensure_merged_tables(conn)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")

    mappings = mapper.get_all_mappings()
    comp_ids = list(mappings.comp_id.unique())

    inserted = 0
    processed = 0

    for comp_id in comp_ids:
        if skip_if_comp_done:
            cur.execute("SELECT 1 FROM merged_spectra WHERE comp_id=? LIMIT 1;", (comp_id,))
            if cur.fetchone():
                processed += 1
                if processed % commit_every == 0:
                    conn.commit()
                continue

        spec_ids = mappings.loc[mappings.comp_id == comp_id, "spec_id"].astype(int).tolist()
        spectra = sdb.get_spectra_by_ids(spec_ids)

        if not spectra:
            processed += 1
            if processed % commit_every == 0:
                conn.commit()
            continue

        # Split by (ionmode, charge) to avoid over-merging
        groups = _split_by_mode_charge(spectra)

        for (ionmode, charge), idxs in groups.items():
            block_spectra = [spectra[i] for i in idxs]

            # Cluster within block
            clusters_local, _ = cluster_block(
                block_spectra,
                sim_score=CosineGreedy(intensity_power=intensity_power),
                threshold=cosine_thr)

            # Merge spectra
            merged_block = get_merged_spectra(block_spectra, clusters_local, mz_tol=mz_tol, min_frac=min_frac)

            # Insert rows
            for cl_local, merged_spec in zip(clusters_local, merged_block):
                # cluster members (global spec_ids)
                global_ids = [spec_ids[idxs[j]] for j in cl_local]
                num_merged = len(global_ids)

                # aggregate metadata across the cluster
                cluster_specs = [block_spectra[j] for j in cl_local]
                agg = _aggregate_metadata_for_cluster(cluster_specs)

                # derive core columns
                ionmode_val = _clean_val(agg.get("ionmode"))  # from first
                charge_val = _safe_meta(merged_spec, "charge", None)  # keep original charge if present
                precursor_mz_val = agg.get("precursor_mz", None)

                mz_arr = np.asarray(merged_spec.peaks.mz, dtype=np.float64)
                it_arr = np.asarray(merged_spec.peaks.intensities, dtype=np.float32)

                cur.execute(
                    """
                    INSERT INTO merged_spectra (
                        comp_id, ionmode, charge, precursor_mz,
                        smiles, inchikey, inchi, name,
                        instrument_type, adduct, collision_energy,
                        num_merged, source_spec_ids, mz, intensities,
                        instrument_type_hist, adduct_hist, collision_energy_hist
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        comp_id,
                        ionmode_val,
                        charge_val,
                        float(precursor_mz_val),

                        _clean_val(agg.get("smiles")),
                        _clean_val(agg.get("inchikey")),
                        _clean_val(agg.get("inchi")),
                        _clean_val(agg.get("name")),

                        _clean_val(agg.get("instrument_type")),
                        _clean_val(agg.get("adduct")),
                        _clean_val(agg.get("collision_energy")),

                        int(num_merged),
                        json.dumps(global_ids),

                        sqlite3.Binary(ndarray_to_blob(mz_arr)),
                        sqlite3.Binary(ndarray_to_blob(it_arr)),

                        agg.get("instrument_type_hist"),
                        agg.get("adduct_hist"),
                        agg.get("collision_energy_hist"),
                    ),
                )
                inserted += 1

        processed += 1
        if processed % commit_every == 0:
            conn.commit()

    conn.commit()
    return inserted
