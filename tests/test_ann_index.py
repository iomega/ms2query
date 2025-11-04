import io
import json
import sqlite3
from typing import List, Tuple

import numpy as np
import pytest
from matchms import Spectrum

from ms2query.database import ANNIndex
from ms2query.database.spectra_merging import ensure_merged_tables

# --- small helpers for array <-> BLOB used in tests (mirrors the production helpers) ---

def _ndarray_to_blob(arr: np.ndarray) -> bytes:
    with io.BytesIO() as f:
        np.save(f, arr, allow_pickle=False)
        return f.getvalue()


@pytest.fixture()
def conn() -> sqlite3.Connection:
    # In-memory DB for tests
    return sqlite3.connect(":memory:")


@pytest.fixture()
def ann(conn) -> ANNIndex:
    # Instantiate with dummy model path; we’ll monkeypatch load_model.
    return ANNIndex(
        conn=conn,
        model_path="dummy_model.pt",
        faiss_metric="ip",
        faiss_factory=None,
        normalize_embeddings=True,
    )


def test_ensure_schema_creates_tables(ann: ANNIndex):
    """Schema should be created with all required columns."""
    ann.ensure_schema()
    cur = ann.conn.cursor()
    # Check both tables exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='merged_spectra';")
    assert cur.fetchone() is not None
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='merged_embeddings';")
    assert cur.fetchone() is not None

    # Check a few critical columns exist
    cur.execute("PRAGMA table_info('merged_spectra');")
    cols = {row[1] for row in cur.fetchall()}
    for required in ("merged_id", "comp_id", "precursor_mz", "mz", "intensities", "num_merged"):
        assert required in cols


def _insert_synthetic_merged_rows(conn: sqlite3.Connection) -> Tuple[int, int]:
    """
    Insert two tiny merged_spectra rows with minimal viable metadata.
    Returns their merged_ids (sqlite autoincrement).
    """
    cur = conn.cursor()
    ensure_merged_tables(conn)

    # Synthetic peaks
    mz1 = np.array([100.0, 150.0, 200.0], dtype=np.float64)
    it1 = np.array([0.2, 0.3, 0.5], dtype=np.float32)

    mz2 = np.array([101.0, 151.0, 201.0], dtype=np.float64)
    it2 = np.array([0.4, 0.1, 0.5], dtype=np.float32)

    base_cols = (
        "comp_id, ionmode, charge, precursor_mz, smiles, inchikey, inchi, name, "
        "instrument_type, adduct, collision_energy, num_merged, source_spec_ids, mz, intensities"
    )
    q = f"INSERT INTO merged_spectra ({base_cols}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"

    cur.execute(
        q,
        (
            "C1", "positive", 1, 300.123, "C(CO)O", "AAAA-BBBB-CCCC", "InChI=1S/...", "Compound A",
            "QTOF", "[M+H]+", "NCE 20", 3, json.dumps([11, 12, 13]),
            sqlite3.Binary(_ndarray_to_blob(mz1)), sqlite3.Binary(_ndarray_to_blob(it1))
        ),
    )
    id1 = cur.lastrowid

    cur.execute(
        q,
        (
            "C2", "positive", 1, 450.5, "CCN(CC)CC", "XXXX-YYYY-ZZZZ", "InChI=1S/...", "Compound B",
            "Orbitrap", "[M+H]+", "NCE 25", 2, json.dumps([21, 22]),
            sqlite3.Binary(_ndarray_to_blob(mz2)), sqlite3.Binary(_ndarray_to_blob(it2))
        ),
    )
    id2 = cur.lastrowid

    conn.commit()
    return id1, id2


def test_compute_embeddings_inserts_rows(ann: ANNIndex, monkeypatch):
    """Embeddings should be computed and written; rerun with only_missing yields 0 new rows."""
    id1, id2 = _insert_synthetic_merged_rows(ann.conn)

    # Monkeypatch model loading and embedding to be deterministic & light.
    class _DummyModel:
        pass

    def fake_load_model(_path):
        return _DummyModel()

    def fake_compute_embedding_array(model, specs: List[Spectrum]) -> np.ndarray:
        # simple deterministic embedding: [precursor_mz, charge, sum(intens), len(peaks)]
        out = []
        for s in specs:
            pmz = float(s.metadata["precursor_mz"])
            charge = float(s.metadata.get("charge") or 0)
            intens_sum = float(np.sum(s.peaks.intensities))
            n_peaks = float(len(s.peaks.mz))
            out.append([pmz, charge, intens_sum, n_peaks])
        return np.asarray(out, dtype=np.float32)

    monkeypatch.setattr("ms2query.database.ann_index.load_model", fake_load_model)
    monkeypatch.setattr("ms2query.database.ann_index.compute_embedding_array", fake_compute_embedding_array)

    inserted = ann.compute_embeddings_to_sqlite(batch_rows=64, only_missing=True)
    assert inserted == 2

    # Confirm rows exist
    cur = ann.conn.cursor()
    cur.execute("SELECT COUNT(1) FROM merged_embeddings;")
    assert cur.fetchone()[0] == 2

    # Re-run with only_missing: should insert 0
    inserted2 = ann.compute_embeddings_to_sqlite(batch_rows=64, only_missing=True)
    assert inserted2 == 0


def test_build_index_and_query(ann: ANNIndex, monkeypatch):
    """Build index from stored embeddings and query it; top-1 should be the intended nearest."""
    id1, id2 = _insert_synthetic_merged_rows(ann.conn)

    # Same monkeypatch as previous test (model + embeddings)
    class _DummyModel:
        pass

    def fake_load_model(_path):
        return _DummyModel()

    def fake_compute_embedding_array(model, specs: List[Spectrum]) -> np.ndarray:
        # Embedding consistent with test_compute_embeddings
        out = []
        for s in specs:
            pmz = float(s.metadata["precursor_mz"])
            charge = float(s.metadata.get("charge") or 0)
            intens_sum = float(np.sum(s.peaks.intensities))
            n_peaks = float(len(s.peaks.mz))
            out.append([pmz, charge, intens_sum, n_peaks])
        return np.asarray(out, dtype=np.float32)

    monkeypatch.setattr("ms2query.database.ann_index.load_model", fake_load_model)
    monkeypatch.setattr("ms2query.database.ann_index.compute_embedding_array", fake_compute_embedding_array)

    # Compute embeddings
    ann.compute_embeddings_to_sqlite(batch_rows=64, only_missing=False)

    # Build FAISS index
    index = ann.build_index()
    assert index.ntotal == 2

    # Prepare a query spectrum that should be closest to the 2nd row (precursor_mz=450.5)
    q_mz = np.array([100.0, 200.0], dtype=np.float32)
    q_it = np.array([0.5, 0.5], dtype=np.float32)
    q_spec = Spectrum(mz=q_mz, intensities=q_it, metadata={"precursor_mz": 450.5, "ionmode": "positive", "charge": 1})

    # Query
    results = ann.query(q_spec, k=2, include_metadata=True, as_dataframe=True)
    assert isinstance(results, list) and len(results) == 1
    df = results[0]
    assert {"rank", "merged_id", "score", "distance", "comp_id", "name"}.issubset(df.columns)

    # Top-1 hit should be the row with precursor_mz=450.5 (id2)
    top1 = df.iloc[0]
    assert int(top1["merged_id"]) == id2
    assert top1["comp_id"] == "C2"
    assert top1["name"] == "Compound B"

    # Scores should be non-increasing by rank
    assert np.all(df["score"].values[:-1] >= df["score"].values[1:])
