from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Optional, Dict, Any
import sqlite3
import numpy as np
import pandas as pd
from matchms import Spectrum
from pathlib import Path

# ------------ helpers ------------

_NUMERIC_FIELDS = {"precursor_mz", "collision_energy"}  # stored as REAL
_TEXT_FIELDS = {
    "ionmode", "smiles", "inchikey", "inchi", "name",
    "instrument_type", "adduct"
}

def _as_float32_bytes(a: np.ndarray) -> bytes:
    if a is None:
        return b""
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return a.tobytes(order="C")

def _from_float32_bytes(b: bytes, n: int) -> np.ndarray:
    # n_peaks defines how many valid values
    arr = np.frombuffer(b, dtype=np.float32, count=n)
    # Make it writeable for downstream use
    return np.array(arr, copy=True)

def _normalize_metadata(md: Dict[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in fields:
        val = md.get(key, None)
        if key in _NUMERIC_FIELDS:
            if val is None or (isinstance(val, float) and (np.isnan(val))):
                out[key] = None
            else:
                try:
                    out[key] = float(val)
                except Exception:
                    out[key] = None
        else:
            out[key] = None if val in (None, "") else str(val)
    return out


# ------------ main class ------------

@dataclass
class SpectralDatabase:
    sqlite_path: str
    metadata_fields: List[str] = field(default_factory=lambda: [
        "precursor_mz", "ionmode", "smiles", "inchikey", "inchi", "name",
        "instrument_type", "adduct", "collision_energy"
    ])
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self):
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # ---------- public API ----------

    def add_spectra(self, spectra: List[Spectrum]) -> List[int]:
        """Add spectra to the database. Returns assigned spec_ids."""
        if not spectra:
            return []

        cur = self._conn.cursor()
        # Bulk-load speed PRAGMAs (safe for single-user/batch ingest)
        cur.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
        """)
        cur.execute("BEGIN")

        spec_ids: List[int] = []
        # Try to use RETURNING (SQLite 3.35+), fallback otherwise
        supports_returning = self._supports_returning()

        sql = (
            "INSERT INTO spectra (mz_blob, intensity_blob, n_peaks, "
            + ", ".join(self.metadata_fields)
            + ") VALUES (?,?,?,?,"
            + ",".join("?" for _ in self.metadata_fields[1:])  # first ? after n_peaks already placed
            + ")"
        )
        # Adjust because above mistakenly adds one extra '?'; correct it:
        # Let's build positions precisely:
        placeholders = ",".join("?" for _ in range(3 + len(self.metadata_fields)))
        sql = f"INSERT INTO spectra (mz_blob, intensity_blob, n_peaks, {', '.join(self.metadata_fields)}) VALUES ({placeholders})"
        if supports_returning:
            sql_ret = sql + " RETURNING spec_id"

        try:
            for sp in spectra:
                # matchms Spectrum exposes peaks as arrays; be robust to attribute names
                mz: Optional[np.ndarray] = getattr(sp, "mz", None)
                intens: Optional[np.ndarray] = getattr(sp, "intensities", None)
                # matchms >=0.20 stores as properties; otherwise: sp.peaks.mz, sp.peaks.intensities
                # TODO: clean up and only focus on newer matchms
                if mz is None or intens is None:
                    peaks = getattr(sp, "peaks", None)
                    if peaks is None:
                        raise ValueError("Spectrum lacks fragments (no mz/intensities).")
                    mz = np.asarray(peaks.mz, dtype=np.float32)
                    intens = np.asarray(peaks.intensities, dtype=np.float32)
                else:
                    mz = np.asarray(mz, dtype=np.float32)
                    intens = np.asarray(intens, dtype=np.float32)

                if mz.shape[0] != intens.shape[0]:
                    raise ValueError("m/z and intensity arrays have different lengths.")

                n = int(mz.shape[0])
                md = getattr(sp, "metadata", {}) or {}
                md_norm = _normalize_metadata(md, self.metadata_fields)

                values = [
                    _as_float32_bytes(mz),
                    _as_float32_bytes(intens),
                    n,
                ] + [md_norm[k] for k in self.metadata_fields]

                if supports_returning:
                    row = cur.execute(sql_ret, values).fetchone()
                    spec_ids.append(int(row[0]))
                else:
                    cur.execute(sql, values)
                    spec_ids.append(cur.lastrowid)

            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

        return spec_ids

    def ids(self) -> List[int]:
        """Return all spec_ids in the database."""
        cur = self._conn.cursor()
        rows = cur.execute("SELECT spec_id FROM spectra").fetchall()
        return [int(row["spec_id"]) for row in rows]

    def get_spectra_by_ids(self, specIDs: List[int]) -> List[Spectrum]:
        """Retrieve full Spectrum objects for given specIDs (order preserved, missing IDs skipped)."""
        rows = self._fetch_rows_by_ids(specIDs, cols="spec_id, mz_blob, intensity_blob, n_peaks, " + ", ".join(self.metadata_fields))
        by_id = {row["spec_id"]: row for row in rows}

        result: List[Spectrum] = []
        for sid in specIDs:
            row = by_id.get(sid)
            if row is None:
                continue
            n = int(row["n_peaks"])
            mz = _from_float32_bytes(row["mz_blob"], n)
            inten = _from_float32_bytes(row["intensity_blob"], n)
            md = {k: row[k] for k in self.metadata_fields}
            md["spec_id"] = sid
            result.append(Spectrum(mz=mz, intensities=inten, metadata=md))
        return result

    def get_fragments_by_ids(self, specIDs: List[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve (mz, intensity) arrays for given specIDs (order preserved, missing IDs skipped)."""
        rows = self._fetch_rows_by_ids(specIDs, cols="spec_id, mz_blob, intensity_blob, n_peaks")
        by_id = {row["spec_id"]: row for row in rows}

        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for sid in specIDs:
            row = by_id.get(sid)
            if row is None:
                continue
            n = int(row["n_peaks"])
            mz = _from_float32_bytes(row["mz_blob"], n)
            inten = _from_float32_bytes(row["intensity_blob"], n)
            out.append((mz, inten))
        return out

    def get_metadata_by_ids(self, specIDs: List[int]) -> pd.DataFrame:
        """Retrieve metadata for given specIDs (order preserved)."""
        cols = ["spec_id"] + self.metadata_fields
        rows = self._fetch_rows_by_ids(specIDs, cols=", ".join(cols))
        df = pd.DataFrame(rows, columns=cols)
        # Preserve caller order
        if not df.empty:
            order = {sid: i for i, sid in enumerate(specIDs)}
            df["__order"] = df["spec_id"].map(order)
            df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
        return df

    def sql_query(self, query: str) -> pd.DataFrame:
        """Run a raw SQL SELECT and return a DataFrame."""
        return pd.read_sql_query(query, self._conn)

    # ---------- internal ----------

    def _supports_returning(self) -> bool:
        try:
            v = self._conn.execute("select sqlite_version()").fetchone()[0]
            major, minor, patch = (int(x) for x in v.split("."))
            return (major, minor, patch) >= (3, 35, 0)
        except Exception:
            return False

    def _fetch_rows_by_ids(self, specIDs: List[int], cols: str) -> List[sqlite3.Row]:
        if not specIDs:
            return []
        placeholders = ",".join("?" for _ in specIDs)
        sql = f"SELECT {cols} FROM spectra WHERE spec_id IN ({placeholders})"
        cur = self._conn.cursor()
        return cur.execute(sql, specIDs).fetchall()

    def _ensure_schema(self):
        cur = self._conn.cursor()
        # Build metadata columns
        md_cols_sql = []
        for k in self.metadata_fields:
            if k in _NUMERIC_FIELDS:
                md_cols_sql.append(f"{k} REAL")
            else:
                md_cols_sql.append(f"{k} TEXT")
        md_cols_clause = ", ".join(md_cols_sql)

        cur.executescript(f"""
            CREATE TABLE IF NOT EXISTS spectra(
                spec_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                mz_blob       BLOB NOT NULL,
                intensity_blob BLOB NOT NULL,
                n_peaks       INTEGER NOT NULL,
                {md_cols_clause}
            );
            CREATE INDEX IF NOT EXISTS idx_inchikey ON spectra(inchikey);
            CREATE INDEX IF NOT EXISTS idx_precursor_mz ON spectra(precursor_mz);
        """)
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
