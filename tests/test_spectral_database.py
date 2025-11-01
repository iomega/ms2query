# test_spectral_database.py
import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum

from ms2query.spectral_database import SpectralDatabase


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "spectra.sqlite"
    db = SpectralDatabase(str(db_path))
    yield db
    db.close()


def make_spectrum(mz, intens, **metadata):
    return Spectrum(
        mz=np.asarray(mz, dtype="float"),
        intensities=np.asarray(intens, dtype="float"),
        metadata=metadata,
    )


@pytest.fixture
def spectra_small():
    s1 = make_spectrum(
        [5, 110, 220, 330, 399, 440],
        [10, 10, 1, 10, 20, 100],
        precursor_mz=240.0,
        ionmode="positive",
    )
    s2 = make_spectrum(
        [50.5, 75.3, 125.0],
        [100, 20, 10],
        precursor_mz=123.4,
        name="test-2",
        instrument_type="Orbitrap",
    )
    s3 = make_spectrum(
        [101, 202, 303, 404],
        [1, 2, 3, 4],
        precursor_mz=404.1,
        inchikey="ABCD-IK",
        collision_energy=35.0,
        adduct="[M+H]+",
    )
    return [s1, s2, s3]


def test_add_and_retrieve_single(tmp_db):
    s = make_spectrum(
        [5, 110, 220, 330, 399, 440],
        [10, 10, 1, 10, 20, 100],
        precursor_mz=240.0,
    )
    ids = tmp_db.add_spectrum([s])
    assert isinstance(ids, list) and len(ids) == 1
    sid = ids[0]
    assert isinstance(sid, int)

    out = tmp_db.retrieve_spectra_by_ids([sid])
    assert len(out) == 1
    sp_out = out[0]

    # values equal (allow float tolerance), dtype is float32 in storage
    assert np.allclose(sp_out.mz, np.array([5, 110, 220, 330, 399, 440], dtype=np.float32))
    assert np.allclose(sp_out.intensities, np.array([10, 10, 1, 10, 20, 100], dtype=np.float32))
    assert sp_out.mz.dtype == np.float32
    assert sp_out.intensities.dtype == np.float32

    # metadata contains what we stored + spec_id
    assert sp_out.metadata.get("precursor_mz") == pytest.approx(240.0)
    assert sp_out.metadata.get("spec_id") == sid


def test_add_and_retrieve_multiple_order_preserved(tmp_db, spectra_small):
    ids = tmp_db.add_spectrum(spectra_small)
    assert len(ids) == 3

    # request in a permuted order; results should follow request order
    req = [ids[2], ids[0], ids[1]]
    out = tmp_db.retrieve_spectra_by_ids(req)
    assert [sp.metadata["spec_id"] for sp in out] == req

    # spot-check one item’s content
    s2 = out[1]  # corresponds to ids[0]
    assert s2.metadata["precursor_mz"] == pytest.approx(240.0)


def test_retrieve_fragments_by_ids(tmp_db, spectra_small):
    ids = tmp_db.add_spectrum(spectra_small)
    req = [ids[1], ids[1], 9999999, ids[0]]  # includes duplicate + missing
    # Implementation skips missing IDs and preserves order for the present ones
    frags = tmp_db.retrieve_fragments_by_ids(req)
    # We expect two results (the duplicate is returned twice; missing is skipped)
    assert len(frags) == 3
    (mz_a, in_a), (mz_b, in_b), (mz_c, in_c) = frags

    # dtype should be float32
    for arr in (mz_a, in_a, mz_b, in_b, mz_c, in_c):
        assert arr.dtype == np.float32

    # shape matches inputs for those spectra
    assert mz_a.shape[0] == spectra_small[1].mz.shape[0]
    assert mz_c.shape[0] == spectra_small[0].mz.shape[0]


def test_retrieve_metadata_by_ids_df(tmp_db, spectra_small):
    ids = tmp_db.add_spectrum(spectra_small)
    df = tmp_db.retrieve_metadata_by_ids([ids[2], ids[0]])

    # Expected columns: spec_id + configured metadata fields
    expected_cols = ["spec_id"] + tmp_db.metadata_fields
    assert list(df.columns) == expected_cols

    # Two rows, in the requested order
    assert df.shape[0] == 2
    assert df.loc[0, "spec_id"] == ids[2]
    assert df.loc[1, "spec_id"] == ids[0]

    # Stored values present / normalized
    assert df.loc[0, "inchikey"] == "ABCD-IK"  # came from spectrum_3
    assert df.loc[1, "precursor_mz"] == pytest.approx(240.0)  # came from spectrum_1

    # Missing fields become None
    assert pd.isna(df.loc[1, "inchikey"]) or df.loc[1, "inchikey"] is None


def test_sql_query_simple(tmp_db, spectra_small):
    ids = tmp_db.add_spectrum(spectra_small)
    df = tmp_db.sql_query("SELECT COUNT(*) AS n FROM spectra")
    assert df.iloc[0]["n"] == 3
    # Query some metadata back
    df2 = tmp_db.sql_query("SELECT spec_id, precursor_mz FROM spectra ORDER BY spec_id")
    assert set(df2.columns) == {"spec_id", "precursor_mz"}
    assert len(df2) == 3


def test_missing_ids_handling(tmp_db, spectra_small):
    ids = tmp_db.add_spectrum(spectra_small)
    req = [999999, ids[1]]
    out_spectra = tmp_db.retrieve_spectra_by_ids(req)
    out_meta = tmp_db.retrieve_metadata_by_ids(req)
    out_frags = tmp_db.retrieve_fragments_by_ids(req)

    # Implementation skips missing IDs but preserves order of the ones that exist
    assert [s.metadata["spec_id"] for s in out_spectra] == [ids[1]]
    assert list(out_meta["spec_id"]) == [ids[1]]
    assert len(out_frags) == 1
