from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any, List, Tuple
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator

from ms2query.fingerprint_computation import compute_fingerprints_from_smiles

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

def encode_sparse_fp(bits: Optional[np.ndarray], counts: Optional[np.ndarray]) -> tuple[bytes, bytes]:
    """Store bits as uint32 indices, counts as int32
    Returns (bits_blob, counts_blob). Accepts None -> empty blobs."""
    if bits is None:
        b = b""
    else:
        arr = np.asarray(bits)
        if arr.dtype != np.uint32:
            arr = arr.astype(np.uint32, copy=False)
        b = arr.tobytes(order="C")
    if counts is None:
        c = b""
    else:
        arrc = np.asarray(counts)
        if arrc.dtype != np.int32 and arrc.dtype != np.uint32 and arrc.dtype != np.uint16 and arrc.dtype != np.uint8:
            arrc = arrc.astype(np.int32, copy=False)
        c = arrc.tobytes(order="C")
    return b, c

def decode_sparse_fp(bits_blob: bytes, counts_blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of encode_sparse_fp. Returns (bits_uint32, counts_int32). Empty blobs -> empty arrays."""
    bits = np.frombuffer(bits_blob, dtype=np.uint32).copy() if bits_blob else np.zeros(0, dtype=np.uint32)
    # Guess signedness: store as int32 by default
    counts = np.frombuffer(counts_blob, dtype=np.int32).copy() if counts_blob else np.zeros(0, dtype=np.int32)
    return bits, counts

def decode_fp_blob(blob: bytes) -> np.ndarray:
    """Decode fingerprint BLOB back to uint8 array. Unknown length -> infer from blob size."""
    if not blob:
        return np.zeros(0, dtype=np.uint8)
    return np.frombuffer(blob, dtype=np.uint8).copy()

def compute_fingerprints(
        smiles: Optional[str] = None,
        inchis: Optional[str] = None,
        sparse: bool = True,
        count: bool = True,
        radius: int = 9,
        progress_bar: bool = True,
        ) -> np.ndarray:
    """
    Placeholder: compute a molecular fingerprint from SMILES or InChI.
    For now return a dummy vector (replace with RDKit/Morgan etc. later).
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=4096)

    if inchis and not smiles:
        # convert inchis to smiles
        smiles = []
        for inchi in inchis:
            try:
                from rdkit import Chem
                mol = Chem.MolFromInchi(inchi)
                smi = Chem.MolToSmiles(mol) if mol is not None else None
                smiles.append(smi)
            except Exception as e:
                print(f"Error converting InChI to SMILES for {inchi}: {e}")
                smiles.append(None)
    elif not smiles and not inchis:
        raise ValueError("Either smiles or inchis must be provided.")
    return compute_fingerprints_from_smiles(
        smiles, 
        fpgen,
        count=count,
        sparse=sparse,
        progress_bar=progress_bar,
    )


# ==================================================
# Compound database (compounds table) in SQLite
# ==================================================

# --- keep your imports/utilities as-is (encode/decode utils etc.) ---

@dataclass
class CompoundDatabase:
    sqlite_path: str
    compound_fields: List[str] = field(default_factory=lambda: [
        "smiles", "inchi", "inchikey", "classyfire_class", "classyfire_superclass"
    ])
    # Default FP parameters (used by the backfill method if you pass through to your compute_fingerprints)
    fingerprint_radius: int = 9
    fingerprint_sparse: bool = True
    fingerprint_count: bool = True
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self):
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self):
        try: self._conn.close()
        except Exception: pass

    def _ensure_schema(self):
        cur = self._conn.cursor()
        cur.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS compounds(
                comp_id               TEXT PRIMARY KEY,          -- inchikey14
                smiles                TEXT,
                inchi                 TEXT,
                inchikey              TEXT UNIQUE,
                -- old single blob may still exist; unused
                fingerprint           BLOB,
                classyfire_class      TEXT,
                classyfire_superclass TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_compounds_smiles ON compounds(smiles);
            CREATE INDEX IF NOT EXISTS idx_compounds_inchi  ON compounds(inchi);
        """)
        # add missing columns for sparse pair
        cols = {r[1] for r in cur.execute("PRAGMA table_info(compounds)").fetchall()}
        for name, typ in (("fingerprint_bits", "BLOB"), ("fingerprint_counts", "BLOB")):
            if name not in cols:
                cur.execute(f"ALTER TABLE compounds ADD COLUMN {name} {typ}")
        self._conn.commit()

    # ---------- UPSERTS: write metadata only; DO NOT compute fingerprints here ----------

    def upsert_compound(
        self,
        smiles: Optional[str] = None,
        inchi: Optional[str] = None,
        inchikey: Optional[str] = None,
        classyfire_class: Optional[str] = None,
        classyfire_superclass: Optional[str] = None,
        fingerprint: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # allowed, but not required
    ) -> str:
        if inchikey is None:
            raise ValueError("inchikey is required to form comp_id (inchikey14).")
        comp_id = inchikey14_from_full(inchikey)
        if not comp_id:
            raise ValueError(f"Invalid InChIKey: {inchikey}")

        # If a fingerprint tuple was explicitly passed, persist it; otherwise leave empty
        if fingerprint is not None:
            bits_blob, counts_blob = encode_sparse_fp(*fingerprint)
        else:
            bits_blob, counts_blob = b"", b""

        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO compounds (
                comp_id, smiles, inchi, inchikey,
                fingerprint_bits, fingerprint_counts,
                classyfire_class, classyfire_superclass
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(comp_id) DO UPDATE SET
                smiles=COALESCE(excluded.smiles, compounds.smiles),
                inchi=COALESCE(excluded.inchi, compounds.inchi),
                inchikey=COALESCE(excluded.inchikey, compounds.inchikey),
                fingerprint_bits=CASE
                    WHEN COALESCE(LENGTH(excluded.fingerprint_bits),0) > 0
                    THEN excluded.fingerprint_bits ELSE compounds.fingerprint_bits END,
                fingerprint_counts=CASE
                    WHEN COALESCE(LENGTH(excluded.fingerprint_counts),0) > 0
                    THEN excluded.fingerprint_counts ELSE compounds.fingerprint_counts END,
                classyfire_class=COALESCE(excluded.classyfire_class, compounds.classyfire_class),
                classyfire_superclass=COALESCE(excluded.classyfire_superclass, compounds.classyfire_superclass)
        """, (
            comp_id, smiles, inchi, inchikey,
            bits_blob, counts_blob,
            classyfire_class, classyfire_superclass,
        ))
        self._conn.commit()
        return comp_id

    def upsert_many(self, rows: Iterable[Dict[str, Any]]) -> List[str]:
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

                # explicit fingerprint tuple allowed; else empty blobs now
                fp = r.get("fingerprint")
                if fp is not None:
                    bits_blob, counts_blob = encode_sparse_fp(*fp)
                else:
                    bits_blob, counts_blob = b"", b""

                cur.execute("""
                    INSERT INTO compounds (
                        comp_id, smiles, inchi, inchikey,
                        fingerprint_bits, fingerprint_counts,
                        classyfire_class, classyfire_superclass
                    ) VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(comp_id) DO UPDATE SET
                        smiles=COALESCE(excluded.smiles, compounds.smiles),
                        inchi=COALESCE(excluded.inchi, compounds.inchi),
                        inchikey=COALESCE(excluded.inchikey, compounds.inchikey),
                        fingerprint_bits=CASE
                            WHEN COALESCE(LENGTH(excluded.fingerprint_bits),0) > 0
                            THEN excluded.fingerprint_bits ELSE compounds.fingerprint_bits END,
                        fingerprint_counts=CASE
                            WHEN COALESCE(LENGTH(excluded.fingerprint_counts),0) > 0
                            THEN excluded.fingerprint_counts ELSE compounds.fingerprint_counts END,
                        classyfire_class=COALESCE(excluded.classyfire_class, compounds.classyfire_class),
                        classyfire_superclass=COALESCE(excluded.classyfire_superclass, compounds.classyfire_superclass)
                """, (
                    comp_id,
                    r.get("smiles"),
                    r.get("inchi"),
                    inchikey,
                    bits_blob, counts_blob,
                    r.get("classyfire_class"),
                    r.get("classyfire_superclass"),
                ))
                comp_ids.append(comp_id)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        return comp_ids

    # ---------- READ ----------
    # ---------- single-row getters ----------

    def get_compound(self, comp_id: str) -> Optional[Dict[str, Any]]:
        """
        Return metadata for one compound (no fingerprint blobs).
        Keys: comp_id, smiles, inchi, inchikey, classyfire_class, classyfire_superclass
        """
        row = self._conn.execute("""
            SELECT comp_id, smiles, inchi, inchikey, classyfire_class, classyfire_superclass
            FROM compounds
            WHERE comp_id = ?
        """, (comp_id,)).fetchone()
        return dict(row) if row else None

    def get_fingerprint(self, comp_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (bits, counts) tuple for one compound; None if absent or empty.
        """
        row = self._conn.execute("""
            SELECT fingerprint_bits, fingerprint_counts
            FROM compounds
            WHERE comp_id = ?
        """, (comp_id,)).fetchone()
        if not row:
            return None
        bits_blob = row["fingerprint_bits"] or b""
        counts_blob = row["fingerprint_counts"] or b""
        if not bits_blob and not counts_blob:
            return None
        return decode_sparse_fp(bits_blob, counts_blob)

    # ---------- batch getters ----------

    def get_compounds(self, comp_ids: List[str]) -> pd.DataFrame:
        """
        Return metadata for many compounds (no fingerprint blobs), order preserved as in comp_ids.
        Missing comp_ids are omitted from the result.
        """
        if not comp_ids:
            return pd.DataFrame(columns=[
                "comp_id", "smiles", "inchi", "inchikey", "classyfire_class", "classyfire_superclass"
            ])
        placeholders = ",".join("?" for _ in comp_ids)
        df = pd.read_sql_query(f"""
            SELECT comp_id, smiles, inchi, inchikey, classyfire_class, classyfire_superclass
            FROM compounds
            WHERE comp_id IN ({placeholders})
        """, self._conn, params=comp_ids)

        if df.empty:
            return df

        # preserve caller order
        order = {cid: i for i, cid in enumerate(comp_ids)}
        df["__order"] = df["comp_id"].map(order)
        df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
        return df

    def get_fingerprints(self, comp_ids: List[str]) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Return a list of fingerprints aligned with comp_ids.
        Each item is (bits, counts) or None if not found/empty.
        """
        if not comp_ids:
            return []

        placeholders = ",".join("?" for _ in comp_ids)
        rows = self._conn.execute(f"""
            SELECT comp_id, fingerprint_bits, fingerprint_counts
            FROM compounds
            WHERE comp_id IN ({placeholders})
        """, comp_ids).fetchall()

        by_id = {
            r["comp_id"]:
                (None if (not (r["fingerprint_bits"] or b"") and not (r["fingerprint_counts"] or b""))
                 else decode_sparse_fp(r["fingerprint_bits"] or b"", r["fingerprint_counts"] or b""))
            for r in rows
        }

        # align with input order; use None for missing
        return [by_id.get(cid) for cid in comp_ids]


    def sql_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self._conn)

    # ---------- Compute fingerprints later, for all missing ----------

    def compute_fingerprints_missing(
        self,
        batch_size: int = 1000,
        use_progress_bar: bool = True,
        fp_size: int = 4096,
        radius: Optional[int] = None,
        sparse: Optional[bool] = None,
        count: Optional[bool] = None,
    ) -> dict:
        """
        Compute fingerprints for all compounds that have SMILES (pass A) or, if no SMILES,
        have InChI (pass B), and where fingerprints are missing.
        Uses the project-level `compute_fingerprints` function that returns a
        List[Optional[Tuple[np.ndarray,np.ndarray]]].

        Returns stats: {"updated": int, "attempted": int, "skipped": int}
        """
        # parameters default to class defaults if not provided
        radius = self.fingerprint_radius if radius is None else radius
        sparse = self.fingerprint_sparse if sparse is None else sparse
        count  = self.fingerprint_count  if count  is None else count

        cur = self._conn.cursor()

        def _select_batch(sql: str, params: tuple) -> List[sqlite3.Row]:
            return cur.execute(sql, params).fetchall()

        # helper: write results back
        def _update_rows(comp_ids: List[str], results: List[Optional[Tuple[np.ndarray, np.ndarray]]]) -> int:
            updated = 0
            cur.execute("BEGIN")
            try:
                for cid, res in zip(comp_ids, results):
                    if res is None:
                        bits_blob, counts_blob = b"", b""
                    else:
                        bits_blob, counts_blob = encode_sparse_fp(*res)
                        updated += 1
                    cur.execute(
                        "UPDATE compounds SET fingerprint_bits=?, fingerprint_counts=? WHERE comp_id=?",
                        (bits_blob, counts_blob, cid),
                    )
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
            return updated

        stats = {"updated": 0, "attempted": 0, "skipped": 0}

        # PASS A: SMILES-present & fingerprints missing
        sql_smiles = """
            SELECT comp_id, smiles
            FROM compounds
            WHERE smiles IS NOT NULL
              AND TRIM(smiles) <> ''
              AND COALESCE(LENGTH(fingerprint_bits),0)=0
              AND COALESCE(LENGTH(fingerprint_counts),0)=0
            LIMIT ?
            OFFSET ?
        """

        # PASS B: no SMILES, but InChI-present & fingerprints missing
        sql_inchi = """
            SELECT comp_id, inchi
            FROM compounds
            WHERE (smiles IS NULL OR TRIM(smiles) = '')
              AND inchi IS NOT NULL
              AND TRIM(inchi) <> ''
              AND COALESCE(LENGTH(fingerprint_bits),0)=0
              AND COALESCE(LENGTH(fingerprint_counts),0)=0
            LIMIT ?
            OFFSET ?
        """

        for sql, which in [(sql_smiles, "smiles"), (sql_inchi, "inchi")]:
            offset = 0
            while True:
                rows = _select_batch(sql, (batch_size, offset))
                if not rows:
                    break
                comp_ids = [r[0] for r in rows]
                reps = [r[1] for r in rows]  # list[str] of smiles or inchi

                # call your project-level function ONCE for the whole batch
                results = compute_fingerprints(
                    smiles=reps if which == "smiles" else None,
                    inchis=reps if which == "inchi" else None,
                    sparse=sparse,
                    count=count,
                    radius=radius,
                    progress_bar=use_progress_bar,
                )  # -> List[Optional[Tuple[np.ndarray,np.ndarray]]]

                upd = _update_rows(comp_ids, results)
                stats["updated"] += upd
                stats["attempted"] += len(comp_ids)
                offset += batch_size

        # rows without SMILES & without InChI are skipped
        stats["skipped"] = self.sql_query("""
            SELECT COUNT(*) AS n
            FROM compounds
            WHERE (smiles IS NULL OR TRIM(smiles)='')
              AND (inchi  IS NULL OR TRIM(inchi) ='')
        """)["n"].iloc[0]

        return stats



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

    # Discover which columns exist in the spectra table
    cols = {r[1] for r in s_conn.execute("PRAGMA table_info(spectra)").fetchall()}
    want = ["spec_id", "inchikey", "smiles", "inchi", "classyfire_class", "classyfire_superclass"]
    have = [c for c in want if c in cols]
    select_cols = ", ".join(have)

    rows = s_conn.execute(f"SELECT {select_cols} FROM spectra").fetchall()

    to_link: List[Tuple[int, str]] = []
    new_comp_rows: List[Dict[str, Any]] = []

    for r in rows:
        r = dict(r)
        spec_id = int(r["spec_id"])
        ik_full = r.get("inchikey")
        if not ik_full:
            continue
        comp_id = inchikey14_from_full(ik_full)
        if not comp_id:
            continue
        to_link.append((spec_id, comp_id))

        if create_missing_compounds:
            new_comp_rows.append({
                "smiles": r.get("smiles"),
                "inchi": r.get("inchi"),
                "inchikey": ik_full,
                "classyfire_class": r.get("classyfire_class"),
                "classyfire_superclass": r.get("classyfire_superclass"),
                "fingerprint": None,  # still defer; backfill later
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
