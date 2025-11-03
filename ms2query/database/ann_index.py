from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import sqlite3
import json
import numpy as np
import pandas as pd
from matchms import Spectrum
from ms2deepscore.models import load_model
from ms2deepscore import compute_embedding_array
import faiss

from .spectra_merging import ensure_merged_tables  # schema with precursor_mz + metadata fields
from .database_utils import blob_to_ndarray, ndarray_to_blob


@dataclass
class ANNIndex:
    """
    End-to-end manager for MS2DeepScore ANN indices backed by SQLite.

    Responsibilities
    ---------------
    - Compute and persist MS2DeepScore embeddings for rows in `merged_spectra`
      into `merged_embeddings`.
    - Build a FAISS index whose IDs are `merged_id`.
    - Query the index and resolve results to rich metadata (and optionally peaks/sources).

    Attributes
    ----------
    conn : sqlite3.Connection
        Connection to the same SQLite DB that contains `merged_spectra` and `merged_embeddings`.
    model_file_name : str
        Path to the MS2DeepScore .pt model.
    faiss_metric:
        Metric the index was built with. If "ip" and `normalize_query=True`, we L2-normalize
        query embeddings (cosine-like behavior).
    faiss_factory:
        Select available faiss factory ( e.g. "IVF4096,Flat", "HNSW32", ...see faiss documentation).
    normalize_embeddings:
        Normalize stored vectors for IP to behave cosine-like. Default is True.
    _model:
        MS2DeepScore model.
    _index:
        ANN index.
        
    Notes
    -----
    - This class is intentionally state-light, it holds a `sqlite3.Connection`, a FAISS index,
      and a lazily-loaded MS2DeepScore model.
    """
    conn: sqlite3.Connection
    model_path: str
    faiss_metric: str = "ip"
    faiss_factory: Optional[str] = None
    normalize_embeddings: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _index: Optional[faiss.Index] = field(default=None, init=False, repr=False)

    # ---------- lifecycle ----------
    def ensure_schema(self) -> None:
        """Create (if absent) the merged tables with the richer schema."""
        ensure_merged_tables(self.conn)

    def load_model(self):
        """Lazy-load MS2DeepScore model."""
        if self._model is None:
            self._model = load_model(self.model_path)
        return self._model

    # ---------- step 2a: embeddings ----------
    def compute_embeddings_to_sqlite(
        self,
        *,
        batch_rows: int = 1024,
        only_missing: bool = True,
        commit_every: int = 0,
    ) -> int:
        """
        Compute MS2DeepScore embeddings for `merged_spectra` and write to `merged_embeddings`.

        Parameters
        ----------
        batch_rows : int
            Number of DB rows to embed at once.
        only_missing : bool
            If True, only embed rows not already in `merged_embeddings`.
        commit_every : int
            Forces a commit after this many merged_ids. 0 means commit only per batch.

        Returns
        -------
        int
            Number of embeddings inserted/updated.
        """
        self.ensure_schema()
        cur = self.conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        if only_missing:
            query = """
                SELECT s.merged_id, s.mz, s.intensities, s.precursor_mz, s.ionmode, s.charge
                FROM merged_spectra s
                LEFT JOIN merged_embeddings e ON s.merged_id = e.merged_id
                WHERE e.merged_id IS NULL
                ORDER BY s.merged_id ASC;
            """
        else:
            query = """
                SELECT merged_id, mz, intensities, precursor_mz, ionmode, charge
                FROM merged_spectra
                ORDER BY merged_id ASC;
            """
        cur.execute(query)

        model = self.load_model()

        inserted = 0
        buf: List[Tuple[int, bytes, bytes, float, str, Optional[int]]] = []
        done_since_commit = 0

        def flush(batch: List[Tuple]) -> int:
            if not batch:
                return 0
            specs: List[Spectrum] = []
            mids: List[int] = []
            for mid, mz_blob, it_blob, prec_mz, ionmode, charge in batch:
                mz = blob_to_ndarray(mz_blob).astype(np.float32, copy=False)
                it = blob_to_ndarray(it_blob).astype(np.float32, copy=False)
                specs.append(Spectrum(mz=mz, intensities=it, metadata={
                    "precursor_mz": float(prec_mz),
                    "ionmode": ionmode,
                    "charge": charge,
                }))
                mids.append(mid)

            emb = compute_embedding_array(model, specs).astype(np.float32, copy=False)
            q = "INSERT OR REPLACE INTO merged_embeddings (merged_id, embedding) VALUES (?, ?);"
            with self.conn:
                for mid, vec in zip(mids, emb):
                    self.conn.execute(q, (mid, sqlite3.Binary(ndarray_to_blob(vec))))
            return len(batch)

        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                break
            buf.extend(rows)
            # process buffer in batch_rows-sized chunks
            while len(buf) >= batch_rows:
                inserted += flush(buf[:batch_rows])
                buf = buf[batch_rows:]
                done_since_commit += batch_rows
                if commit_every and done_since_commit >= commit_every:
                    self.conn.commit()
                    done_since_commit = 0

        inserted += flush(buf)
        return inserted

    # ---------- step 2b: build faiss ----------
    def build_index(self, *, index_path: Optional[str] = None) -> faiss.Index:
        """
        Build a FAISS index from `merged_embeddings`. Uses IndexIDMap2 (ids=merged_id).
        Optionally saves to disk.

        Returns
        -------
        faiss.Index
        """
        cur = self.conn.cursor()
        cur.execute("SELECT merged_id, embedding FROM merged_embeddings ORDER BY merged_id ASC;")

        first = cur.fetchone()
        if not first:
            raise ValueError("No embeddings present in 'merged_embeddings'.")
        first_id, first_blob = first
        first_vec = blob_to_ndarray(first_blob).astype(np.float32, copy=False)
        d = int(first_vec.shape[-1])

        metric = faiss.METRIC_INNER_PRODUCT if self.faiss_metric.lower() == "ip" else faiss.METRIC_L2
        base = (faiss.index_factory(d, self.faiss_factory, metric)
                if self.faiss_factory else
                (faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)))
        index = faiss.IndexIDMap2(base)

        # add first vector
        X = first_vec[None, :]
        if self.faiss_metric.lower() == "ip" and self.normalize_embeddings:
            faiss.normalize_L2(X)
        index.add_with_ids(X, np.array([first_id], dtype=np.int64))

        # stream the rest
        BATCH = 8192
        ids_buf, vec_buf = [], []
        while True:
            rows = cur.fetchmany(BATCH)
            if not rows:
                break
            for mid, blob in rows:
                ids_buf.append(mid)
                vec_buf.append(blob_to_ndarray(blob))
            if ids_buf:
                Xb = np.vstack(vec_buf).astype(np.float32, copy=False)
                if self.faiss_metric.lower() == "ip" and self.normalize_embeddings:
                    faiss.normalize_L2(Xb)
                index.add_with_ids(Xb, np.asarray(ids_buf, dtype=np.int64))
                ids_buf.clear()
                vec_buf.clear()

        if index_path:
            faiss.write_index(index, index_path)

        self._index = index
        return index

    def load_index(self, index_path: str) -> faiss.Index:
        """Load an existing FAISS index from disk."""
        self._index = faiss.read_index(index_path)
        return self._index

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("FAISS index not built/loaded. Call build_index() or load_index().")
        return self._index

    # ---------- querying ----------
    def _fetch_merged_rows(
        self,
        ids: List[int],
        *,
        include_peaks: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        """Bulk-fetch rows from merged_spectra for a list of merged_id values."""
        if not ids:
            return {}
        out: Dict[int, Dict[str, Any]] = {}
        CHUNK = 1000
        q_base = (
            "SELECT merged_id, comp_id, ionmode, charge, precursor_mz, "
            "smiles, inchikey, inchi, name, instrument_type, adduct, collision_energy, "
            "num_merged, source_spec_ids, mz, intensities "
            "FROM merged_spectra WHERE merged_id IN ({ph});"
        )
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i+CHUNK]
            q = q_base.format(ph=",".join("?" for _ in chunk))
            rows = self.conn.execute(q, chunk).fetchall()
            for r in rows:
                (mid, comp_id, ionmode, charge, precursor_mz, smiles, inchikey, inchi, name,
                 instrument_type, adduct, collision_energy, num_merged, source_spec_ids,
                 mz_blob, intens_blob) = r
                row = {
                    "merged_id": mid, "comp_id": comp_id, "ionmode": ionmode, "charge": charge,
                    "precursor_mz": precursor_mz, "smiles": smiles, "inchikey": inchikey,
                    "inchi": inchi, "name": name, "instrument_type": instrument_type,
                    "adduct": adduct, "collision_energy": collision_energy, "num_merged": num_merged,
                    "source_spec_ids": json.loads(source_spec_ids) if source_spec_ids else [],
                }
                if include_peaks:
                    row["mz"] = blob_to_ndarray(mz_blob).astype(np.float32, copy=False)
                    row["intensities"] = blob_to_ndarray(intens_blob).astype(np.float32, copy=False)
                out[mid] = row
        return out

    def query(
        self,
        queries: Union[Spectrum, List[Spectrum]],
        *,
        k: int = 10,
        include_metadata: bool = True,
        include_peaks: bool = False,
        include_sources: bool = False,
        sdb: Optional[Any] = None,
        as_dataframe: bool = True,
    ) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:
        """
        Embed query spectrum(ae), search FAISS, and resolve hits via SQLite.

        Returns per-query results (DataFrame by default) with columns:
        ['rank','merged_id','score','distance', 'comp_id','name','ionmode','charge',
         'precursor_mz','adduct','collision_energy','num_merged','source_spec_ids', ...]
        (plus 'merged_spectrum' / 'source_spectra' if requested).

        Notes
        -----
        - For `faiss_metric="ip"` with normalized vectors, 'score' is cosine-like.
        - For `faiss_metric="l2"`, 'score' is just `-distance` (so “higher is better”).
        """
        if isinstance(queries, Spectrum):
            queries = [queries]

        model = self.load_model()
        Q = compute_embedding_array(model, queries).astype(np.float32, copy=False)

        if self.faiss_metric.lower() == "ip" and self.normalize_embeddings:
            faiss.normalize_L2(Q)

        distances, ids = self.index.search(Q, k)  # (nq, k)
        nq = distances.shape[0]
        flat_ids = [int(x) for x in ids.flatten().tolist() if x != -1]

        rows_by_id: Dict[int, Dict[str, Any]] = {}
        if include_metadata or include_peaks or include_sources:
            rows_by_id = self._fetch_merged_rows(flat_ids, include_peaks=include_peaks)

        results_all: List[List[Dict[str, Any]]] = []
        for qi in range(nq):
            one: List[Dict[str, Any]] = []
            for rk in range(k):
                mid = int(ids[qi, rk])
                if mid == -1:
                    continue
                dist = float(distances[qi, rk])
                score = dist if self.faiss_metric.lower() == "ip" else -dist
                item: Dict[str, Any] = {"rank": rk+1, "merged_id": mid, "score": score, "distance": dist}

                if include_metadata or include_peaks or include_sources:
                    row = rows_by_id.get(mid)
                    if row:
                        item.update({k: v for k, v in row.items() if k not in ("mz", "intensities")})
                        if include_peaks:
                            mz = row.get("mz")
                            it = row.get("intensities")
                            item["merged_spectrum"] = Spectrum(mz=mz, intensities=it, metadata={
                                "precursor_mz": row["precursor_mz"],
                                "ionmode": row["ionmode"],
                                "charge": row["charge"],
                                "comp_id": row["comp_id"],
                                "num_merged": row["num_merged"],
                                "source_spec_ids": row["source_spec_ids"],
                                "name": row.get("name"),
                                "adduct": row.get("adduct"),
                            }) if (mz is not None and it is not None) else None
                        if include_sources:
                            if sdb is None:
                                raise ValueError("include_sources=True requires sdb.")
                            item["source_spectra"] = sdb.get_spectra_by_ids(row.get("source_spec_ids", []))
                    else:
                        # keep columns consistent
                        if include_peaks:
                            item["merged_spectrum"] = None
                        if include_sources:
                            item["source_spectra"] = []

                one.append(item)
            results_all.append(one)

        if not as_dataframe:
            return results_all

        # convert to tidy DataFrames with a nice column order
        dfs: List[pd.DataFrame] = []
        base_order = [
            "rank", "merged_id", "score", "distance", "comp_id", "name",
            "ionmode", "charge", "precursor_mz", "adduct", "collision_energy",
            "num_merged", "source_spec_ids"
        ]
        for one in results_all:
            df = pd.DataFrame(one)
            # move known columns up front
            cols_front = [c for c in base_order if c in df.columns]
            cols_rest = [c for c in df.columns if c not in cols_front]
            df = df[cols_front + cols_rest]
            dfs.append(df)
        return dfs
