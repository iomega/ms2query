from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any, List, Tuple
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

# =========================
# Utilities & placeholders
# =========================

def inchikey14_from_full(inchikey: str) -> Optional[str]:
    """Return the first 14 characters (inchikey14). Robust to hyphens/malformed keys."""
    if not inchikey:
        return None
    s = str(inchikey).strip().upper()
    if "-" in s:
        return s.split("-", 1)[0][:14]
    return s[:14] if len(s) >= 14 else None

def encode_fp_blob(fp: Optional[np.ndarray]) -> bytes:
    """Encode fingerprint as a uint8 BLOB. Accepts any numeric dtype -> coerces to uint8."""
    if fp is None:
        return b""
    fp = np.asarray(fp)
    if fp.dtype != np.uint8:
        fp = fp.astype(np.uint8, copy=False)
    return fp.tobytes(order="C")

def decode_fp_blob(blob: bytes) -> np.ndarray:
    """Decode fingerprint BLOB back to uint8 array. Unknown length -> infer from blob size."""
    if not blob:
        return np.zeros(0, dtype=np.uint8)
    return np.frombuffer(blob, dtype=np.uint8).copy()

def compute_fingerprints(smiles: Optional[str], inchi: Optional[str]) -> np.ndarray:
    """
    Placeholder: compute a molecular fingerprint from SMILES or InChI.
    For now return a dummy vector (replace with RDKit/Morgan etc. later).
    """
    return np.array([0, 1, 0, 1], dtype=np.uint8)

# ==================================================
# Compound database (compounds table) in SQLite
# ==================================================

@dataclass
class CompoundDatabase:
    sqlite_path: str
    # Extend as needed (add more classyfire-like fields here)
    compound_fields: List[str] = field(default_factory=lambda: [
        "smiles", "inchi", "inchikey", "classyfire_class", "classyfire_superclass"
    ])
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self):
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    # ---------- schema ----------

    def _ensure_schema(self):
        cur = self._conn.cursor()
        # comp_id is inchikey14 (PRIMARY KEY). inchikey (full) must be unique as well if present.
        cur.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS compounds(
                comp_id              TEXT PRIMARY KEY,          -- inchikey14
                smiles               TEXT,
                inchi                TEXT,
                inchikey             TEXT UNIQUE,               -- full InChIKey (27 chars)
                fingerprint          BLOB,                      -- uint8 array
                classyfire_class     TEXT,
                classyfire_superclass TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_compounds_smiles ON compounds(smiles);
            CREATE INDEX IF NOT EXISTS idx_compounds_inchi  ON compounds(inchi);
        """)
        self._conn.commit()

    # ---------- upsert ----------

    def upsert_compound(
        self,
        smiles: Optional[str] = None,
        inchi: Optional[str] = None,
        inchikey: Optional[str] = None,
        classyfire_class: Optional[str] = None,
        classyfire_superclass: Optional[str] = None,
        fingerprint: Optional[np.ndarray] = None,
    ) -> str:
        """Upsert a single compound. Returns comp_id (inchikey14)."""
        if inchikey is None:
            raise ValueError("inchikey is required to form comp_id (inchikey14).")
        comp_id = inchikey14_from_full(inchikey)
        if not comp_id:
            raise ValueError(f"Invalid InChIKey: {inchikey}")

        fp_blob = encode_fp_blob(fingerprint if fingerprint is not None else compute_fingerprints(smiles, inchi))

        cur = self._conn.cursor()
        # Use INSERT ON CONFLICT for upsert semantics
        cur.execute("""
            INSERT INTO compounds (comp_id, smiles, inchi, inchikey, fingerprint, classyfire_class, classyfire_superclass)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(comp_id) DO UPDATE SET
                smiles=excluded.smiles,
                inchi=excluded.inchi,
                inchikey=excluded.inchikey,
                fingerprint=excluded.fingerprint,
                classyfire_class=excluded.classyfire_class,
                classyfire_superclass=excluded.classyfire_superclass
        """, (comp_id, smiles, inchi, inchikey, fp_blob, classyfire_class, classyfire_superclass))
        self._conn.commit()
        return comp_id

    def upsert_many(self, rows: Iterable[Dict[str, Any]]) -> List[str]:
        """
        Upsert many compounds. Each row may include:
        smiles, inchi, inchikey (required), classyfire_class, classyfire_superclass, fingerprint (np.ndarray optional).
        Returns list of comp_ids.
        """
        comp_ids: List[str] = []
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            for r in rows:
                inchikey = r.get("inchikey")
                if not inchikey:
                    raise ValueError("Each row must contain 'inchikey'.")
                comp_id = inchikey14_from_full(inchikey)
                if not comp_id:
                    raise ValueError(f"Invalid InChIKey: {inchikey}")

                smiles = r.get("smiles")
                inchi = r.get("inchi")
                fingerprint = r.get("fingerprint")
                fp_blob = encode_fp_blob(fingerprint if fingerprint is not None else compute_fingerprints(smiles, inchi))
                classyfire_class = r.get("classyfire_class")
                classyfire_superclass = r.get("classyfire_superclass")

                cur.execute("""
                    INSERT INTO compounds (comp_id, smiles, inchi, inchikey, fingerprint, classyfire_class, classyfire_superclass)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(comp_id) DO UPDATE SET
                        smiles=excluded.smiles,
                        inchi=excluded.inchi,
                        inchikey=excluded.inchikey,
                        fingerprint=excluded.fingerprint,
                        classyfire_class=excluded.classyfire_class,
                        classyfire_superclass=excluded.classyfire_superclass
                """, (comp_id, smiles, inchi, inchikey, fp_blob, classyfire_class, classyfire_superclass))
                comp_ids.append(comp_id)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        return comp_ids

    # ---------- queries ----------

    def get_compound(self, comp_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM compounds WHERE comp_id = ?", (comp_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["fingerprint"] = decode_fp_blob(d["fingerprint"])
        return d

    def sql_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self._conn)


# ==================================================
# Mapping: spectrum <-> compound (spec_to_comp)
# ==================================================

@dataclass
class SpecToCompoundMap:
    """Stores (spec_id -> comp_id) mappings in SQLite. Use the SAME DB file as SpectralDatabase for simplicity."""
    sqlite_path: str
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self):
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def _ensure_schema(self):
        cur = self._conn.cursor()
        # No strict FK enforcement (SpectralDatabase may have been created without FK pragma),
        # here: index both sides for fast lookup.
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS spec_to_comp(
                spec_id INTEGER NOT NULL,
                comp_id TEXT    NOT NULL,
                PRIMARY KEY (spec_id),
                -- implicit: comp_id should exist in compounds.comp_id (not enforced here)
                -- to enforce FK, you can enable PRAGMA foreign_keys=ON and create a FK to compounds(comp_id)
                -- if both tables are in the same SQLite file.
                CHECK (length(comp_id) = 14)
            );
            CREATE INDEX IF NOT EXISTS idx_spec_to_comp_comp ON spec_to_comp(comp_id);
        """)
        self._conn.commit()

    # ---------- API ----------

    def link(self, spec_id: int, comp_id: str):
        """Insert or replace a single mapping."""
        if not comp_id or len(comp_id) != 14:
            raise ValueError("comp_id must be inchikey14 (14 characters).")
        self._conn.execute("""
            INSERT INTO spec_to_comp (spec_id, comp_id)
            VALUES (?, ?)
            ON CONFLICT(spec_id) DO UPDATE SET comp_id=excluded.comp_id
        """, (spec_id, comp_id))
        self._conn.commit()

    def link_many(self, pairs: Iterable[Tuple[int, str]]):
        """Bulk link (spec_id, comp_id)."""
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            cur.executemany("""
                INSERT INTO spec_to_comp (spec_id, comp_id)
                VALUES (?, ?)
                ON CONFLICT(spec_id) DO UPDATE SET comp_id=excluded.comp_id
            """, list(pairs))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def get_comp_id_for_specs(self, spec_ids: List[int]) -> pd.DataFrame:
        """Return a DataFrame with columns [spec_id, comp_id] for the provided spec_ids."""
        if not spec_ids:
            return pd.DataFrame(columns=["spec_id", "comp_id"])
        placeholders = ",".join("?" * len(spec_ids))
        rows = self._conn.execute(
            f"SELECT spec_id, comp_id FROM spec_to_comp WHERE spec_id IN ({placeholders})",
            spec_ids
        ).fetchall()
        return pd.DataFrame(rows, columns=["spec_id", "comp_id"])

    def get_specs_for_comp(self, comp_id: str) -> List[int]:
        """Return list of spec_ids for a given comp_id."""
        rows = self._conn.execute("SELECT spec_id FROM spec_to_comp WHERE comp_id = ?", (comp_id,)).fetchall()
        return [r[0] for r in rows]


# ==================================================
# Integrations with SpectralDatabase
# ==================================================

def map_from_spectraldb_metadata(
    spectral_db_sqlite_path: str,
    mapping_sqlite_path: Optional[str] = None,
    compounds_sqlite_path: Optional[str] = None,
    *,
    create_missing_compounds: bool = True
) -> Tuple[int, int]:
    """
    Read spectra metadata (expects 'inchikey' in metadata), create comp_id (inchikey14),
    populate spec_to_comp, and optionally upsert minimal compounds.

    Returns: (n_mapped, n_new_compounds)
    """
    # We do not import the class to avoid circular imports; use sqlite directly.
    s_conn = sqlite3.connect(spectral_db_sqlite_path)
    s_conn.row_factory = sqlite3.Row

    map_db_path = mapping_sqlite_path or spectral_db_sqlite_path
    c_db_path   = compounds_sqlite_path or spectral_db_sqlite_path

    mapper = SpecToCompoundMap(map_db_path)
    compdb = CompoundDatabase(c_db_path)

    # Pull spec_id + inchikey from SpectralDatabase.spectra table
    # (the earlier SpectralDatabase example stores metadata as columns; ensure 'inchikey' exists there).
    rows = s_conn.execute("SELECT spec_id, inchikey FROM spectra").fetchall()

    to_link: List[Tuple[int, str]] = []
    new_comp_rows: List[Dict[str, Any]] = []

    for r in rows:
        spec_id = int(r["spec_id"])
        ik_full = r["inchikey"]
        if not ik_full:
            continue
        comp_id = inchikey14_from_full(ik_full)
        if not comp_id:
            continue
        to_link.append((spec_id, comp_id))

        if create_missing_compounds:
            new_comp_rows.append({
                "smiles": None,
                "inchi": None,
                "inchikey": ik_full,
                "classyfire_class": None,
                "classyfire_superclass": None,
                "fingerprint": None,   # will be replaced by compute_fingerprints(...)
            })

    # Bulk linking
    if to_link:
        mapper.link_many(to_link)

    # Upsert compounds
    n_new_compounds = 0
    if create_missing_compounds and new_comp_rows:
        # Deduplicate by comp_id to avoid redundant upserts
        seen: set[str] = set()
        dedup_rows: List[Dict[str, Any]] = []
        for r in new_comp_rows:
            cid = inchikey14_from_full(r["inchikey"])
            if cid and cid not in seen:
                seen.add(cid)
                dedup_rows.append(r)
        before = compdb.sql_query("SELECT COUNT(*) AS n FROM compounds")["n"].iloc[0]
        compdb.upsert_many(dedup_rows)
        after  = compdb.sql_query("SELECT COUNT(*) AS n FROM compounds")["n"].iloc[0]
        n_new_compounds = int(after - before)

    n_mapped = len(to_link)

    # tidy
    mapper.close()
    compdb.close()
    s_conn.close()

    return n_mapped, n_new_compounds


def get_unique_compounds_from_spectraldb(
    spectral_db_sqlite_path: str,
    external_meta: Optional[pd.DataFrame] = None,
    external_key_col: str = "inchikey14"
) -> pd.DataFrame:
    """
    Return a DataFrame of unique compounds present in the spectral DB, inferred via inchikey → inchikey14.
    Columns: inchikey14, inchikey (full), n_spectra. If `external_meta` is provided,
    it will be left-joined on `external_key_col` (default 'inchikey14').
    """
    conn = sqlite3.connect(spectral_db_sqlite_path)
    conn.row_factory = sqlite3.Row

    # pull spec_id + inchikey from spectra
    df = pd.read_sql_query("SELECT spec_id, inchikey FROM spectra", conn)
    conn.close()

    if df.empty:
        base = pd.DataFrame(columns=["inchikey14", "inchikey", "n_spectra"])
        if external_meta is not None:
            return base.merge(external_meta, how="left", left_on="inchikey14", right_on=external_key_col)
        return base

    # Compute inchikey14
    ik14 = df["inchikey"].fillna("").map(inchikey14_from_full)
    df["inchikey14"] = ik14

    # Aggregate
    agg = (
        df.dropna(subset=["inchikey14"])
          .groupby(["inchikey14"], as_index=False)
          .agg(n_spectra=("spec_id", "count"),
               inchikey=("inchikey", "first"))  # first full key seen
    )

    # Optional join with external meta
    if external_meta is not None and not external_meta.empty:
        agg = agg.merge(external_meta, how="left", left_on="inchikey14", right_on=external_key_col)

    # Order by prevalence
    agg = agg.sort_values("n_spectra", ascending=False).reset_index(drop=True)
    return agg
