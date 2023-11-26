"""Tests load from sqlite functions

These functions are functions to retrieve information from the sqlite database.
"""

import pandas as pd
from ms2query.utils import column_names_for_output


def test_get_metadata_from_sqlite(sqlite_library):
    spectra_id_list = [0, 1]

    result = sqlite_library.get_metadata_from_sqlite(
        spectra_id_list,
        spectrum_id_storage_name="spectrumid")
    assert isinstance(result, dict), "Expected dictionary as output"
    assert len(result) == len(spectra_id_list), \
        "Expected the same number of results as the spectra_id_list"
    for spectrum_id in spectra_id_list:
        assert spectrum_id in result.keys(), \
            f"The spectrum_id {spectrum_id} was expected as key"
        metadata = result[spectrum_id]
        assert isinstance(metadata, dict), \
            "Expected metadata to be stored as dict"
        for key in metadata.keys():
            assert isinstance(key, str), \
                "Expected keys of metadata to be string"
            assert isinstance(metadata[key], (str, float, int, list, tuple)), \
                f"Expected values of metadata to be string {metadata[key]}"


def test_get_ionization_mode_library(sqlite_library):
    ionization_mode = sqlite_library.get_ionization_mode_library()
    assert ionization_mode == "positive"


def test_get_classes_inchikeys(sqlite_library):
    test_inchikeys = ["IYDKWWDUBYWQGF", "KNGPFNUOXXLKCN"]
    classes = sqlite_library.get_classes_inchikeys(test_inchikeys)
    expected_classes = pd.DataFrame([["IYDKWWDUBYWQGF", "b", "c", "d", "e", "f", "g", "h", "i"],
                                     ["KNGPFNUOXXLKCN", "b", "c", "d", "e", "f", "g", "h", "i"]],
                                    columns=["inchikey"] + column_names_for_output(return_non_classifier_columns=False,
                                                                                   return_classifier_columns=True))
    pd.testing.assert_frame_equal(expected_classes, classes)


def test_contains_class_annotiation(sqlite_library):
    assert sqlite_library.contains_class_annotation(), "contains_class_annotation is expected to return True"
