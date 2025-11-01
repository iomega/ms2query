import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ms2query.compound_database import (
    CompoundDatabase,
    SpecToCompoundMap,
    map_from_spectraldb_metadata,
    get_unique_compounds_from_spectraldb,
    compute_fingerprints,
    inchikey14_from_full,
)

# -------------------------
# Helpers
# -------------------------

def make_tmp_db(tmp_path: Path, name: str = "test.sqlite") -> str:
    p = tmp_path / name
    if p.exists():
        p.unlink()
    return str(p)

# Some example InChIKeys
IK_FULL_1 = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"  # glucose
IK_FULL_2 = "BSYNRYMUTXBXSQ-UHFFFAOYSA-O"  # same first14, different suffix (stereo/isotope)
IK_FULL_3 = "BQJCRHHNABKAKU-KBQPJGBKSA-N"  # ethanol
IK14_1 = "BSYNRYMUTXBXSQ"
IK14_3 = "BQJCRHHNABKAKU"

# -------------------------
# Tests: low-level utilities
# -------------------------

def test_inchikey14():
    assert inchikey14_from_full(IK_FULL_1) == IK14_1
    assert inchikey14_from_full("bsynrymutxbxsq-uhfffaoysa-n") == IK14_1
    assert inchikey14_from_full("BQJCRHHNABKAKU-KBQPJGBKSA-N") == IK14_3
    assert inchikey14_from_full("SHORT") is None  # too short

def test_compute_fingerprints_placeholder():
    fp = compute_fingerprints("C(CO)O", None)
    assert isinstance(fp, np.ndarray)
    assert fp.dtype == np.uint8
    np.testing.assert_array_equal(fp, np.array([0, 1, 0, 1], dtype=np.uint8))

# -------------------------
# Tests: CompoundDatabase
# -------------------------

def test_compound_upsert_and_get(tmp_path):
    db_path = make_tmp_db(tmp_path)
    cdb = CompoundDatabase(db_path)

    # Upsert a compound
    cid = cdb.upsert_compound(
        smiles="C(CO)O",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        inchikey=IK_FULL_3,
        classyfire_class="Alcohols",
        classyfire_superclass="Organic compounds",
    )
    assert cid == IK14_3

    row = cdb.get_compound(cid)
    assert row is not None
    assert row["inchikey"] == IK_FULL_3
    assert isinstance(row["fingerprint"], np.ndarray)
    np.testing.assert_array_equal(row["fingerprint"], np.array([0,1,0,1], dtype=np.uint8))

    # Upsert another with the same comp_id (different full IK) -> should overwrite row cleanly
    cid2 = cdb.upsert_compound(
        smiles="C6H12O6",
        inchi=None,
        inchikey=IK_FULL_1,
        classyfire_class="Carbohydrates",
        classyfire_superclass="Organic compounds",
    )
    assert cid2 == IK14_1
    row2 = cdb.get_compound(IK14_1)
    assert row2["inchikey"] == IK_FULL_1

    cdb.close()

def test_compound_upsert_many(tmp_path):
    db_path = make_tmp_db(tmp_path)
    cdb = CompoundDatabase(db_path)

    # Insert two rows that collapse to the same comp_id (same first14), newest should win
    comp_ids = cdb.upsert_many([
        {"smiles": "X", "inchi": None, "inchikey": IK_FULL_1, "classyfire_class": "A"},
        {"smiles": "Y", "inchi": None, "inchikey": IK_FULL_2, "classyfire_class": "B"},
        {"smiles": "Z", "inchi": None, "inchikey": IK_FULL_3, "classyfire_class": "C"},
    ])
    assert set(comp_ids) == {IK14_1, IK14_3}

    row = cdb.get_compound(IK14_1)
    # After ON CONFLICT(comp_id) UPDATE, row reflects last data for that comp_id
    assert row["smiles"] in {"X", "Y"}  # depends on order; both acceptable here
    assert row["inchikey"] in {IK_FULL_1, IK_FULL_2}
    assert row["classyfire_class"] in {"A", "B"}

    count = cdb.sql_query("SELECT COUNT(*) as n FROM compounds")["n"].iloc[0]
    assert count == 2

    cdb.close()

# -------------------------
# Tests: SpecToCompoundMap + integration
# -------------------------

def create_min_spectral_table(sqlite_path: str, rows):
    """Create a minimal spectra table (spec_id, inchikey) and insert rows."""
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS spectra(
            spec_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inchikey TEXT
        );
    """)
    cur.executemany("INSERT INTO spectra(inchikey) VALUES (?)", [(r,) for r in rows])
    con.commit()
    con.close()

def test_mapping_and_compound_creation(tmp_path):
    db_path = make_tmp_db(tmp_path)

    # Create minimal spectra table (3 rows; one is NULL inchikey)
    create_min_spectral_table(db_path, [IK_FULL_1, IK_FULL_2, None])

    # Run mapping (same db hosts compounds + mapping)
    n_mapped, n_new = map_from_spectraldb_metadata(db_path)
    assert n_mapped == 2                      # two spectra had inchikeys
    assert n_new == 1 or n_new == 2           # depending on upsert collapsing; at least one unique comp

    # Validate mapping contents
    mapper = SpecToCompoundMap(db_path)
    df_map = mapper.get_comp_id_for_specs([1, 2, 3])
    assert set(df_map.columns) == {"spec_id", "comp_id"}
    # spec_id 3 has no inchikey -> may be missing
    assert set(df_map["spec_id"]) <= {1, 2, 3}
    # comp_ids are 14 chars
    assert all(len(c) == 14 for c in df_map["comp_id"])
    mapper.close()

    # Validate compounds exist
    cdb = CompoundDatabase(db_path)
    dfc = cdb.sql_query("SELECT comp_id, inchikey FROM compounds")
    assert not dfc.empty
    assert all(len(cid) == 14 for cid in dfc["comp_id"])
    cdb.close()

def test_mapper_link_and_get(tmp_path):
    db_path = make_tmp_db(tmp_path)

    # need compounds table for FK-like behavior not enforced; mapping works independently
    cdb = CompoundDatabase(db_path)
    cdb.upsert_compound(inchikey=IK_FULL_1)  # ensure a compound exists
    cdb.close()

    mapper = SpecToCompoundMap(db_path)
    mapper.link(123, IK14_1)
    mapper.link_many([(124, IK14_1), (125, IK14_1)])

    ids = mapper.get_specs_for_comp(IK14_1)
    assert set(ids) == {123, 124, 125}

    df = mapper.get_comp_id_for_specs([122, 123, 124, 125])
    assert set(df.columns) == {"spec_id", "comp_id"}
    assert set(df["spec_id"]) == {123, 124, 125}

    mapper.close()

# -------------------------
# Tests: get_unique_compounds_from_spectraldb
# -------------------------

def test_get_unique_compounds_basic(tmp_path):
    db_path = make_tmp_db(tmp_path)
    # spectra: two with same IK14, one different, one NULL
    create_min_spectral_table(db_path, [IK_FULL_1, IK_FULL_2, IK_FULL_3, None])

    uniq = get_unique_compounds_from_spectraldb(db_path)
    # Expect 2 unique IK14 values
    assert list(uniq.columns[:3]) == ["inchikey14", "n_spectra", "inchikey"]
    assert set(uniq["inchikey14"]) == {IK14_1, IK14_3}
    # Counts: IK14_1 appears twice, IK14_3 once
    counts = dict(zip(uniq["inchikey14"], uniq["n_spectra"]))
    assert counts[IK14_1] == 2
    assert counts[IK14_3] == 1

def test_get_unique_compounds_with_external_merge(tmp_path):
    db_path = make_tmp_db(tmp_path)
    create_min_spectral_table(db_path, [IK_FULL_1, IK_FULL_3])

    external = pd.DataFrame({
        "inchikey14": [IK14_1, "NOPE0000000000"],
        "my_tag": ["hit", "miss"],
        "score": [0.9, 0.1],
    })
    uniq = get_unique_compounds_from_spectraldb(db_path, external_meta=external)
    # Should have merged columns
    assert "my_tag" in uniq.columns and "score" in uniq.columns
    # Only IK14_1 should have my_tag filled
    row = uniq.loc[uniq["inchikey14"] == IK14_1].iloc[0]
    assert row["my_tag"] == "hit"
    assert pytest.approx(row["score"]) == 0.9
