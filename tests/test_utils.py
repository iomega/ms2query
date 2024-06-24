import os
from io import StringIO
from typing import List
import pandas as pd
import pytest
from matchms import Spectrum
from ms2query.utils import (add_unknown_charges_to_spectra,
                            load_matchms_spectrum_objects_from_file)


def test_convert_files_to_matchms_spectrum_objects_unknown_file(tmp_path):
    """Tests if unknown file raises an Assertion error"""
    with pytest.raises(AssertionError):
        load_matchms_spectrum_objects_from_file(os.path.join(tmp_path, "file_that_does_not_exist.json"))


def test_add_unknown_charges_to_spectra(hundred_test_spectra):
    spectra = hundred_test_spectra
    # Set charges to predefined values
    for spectrum in spectra[:10]:
        spectrum.set("charge", None)
    for spectrum in spectra[10:20]:
        spectrum.set("charge", 1)
    for spectrum in spectra[20:30]:
        spectrum.set("charge", -1)
    for spectrum in spectra[30:]:
        spectrum.set("charge", 2)

    spectra_with_charge = add_unknown_charges_to_spectra(spectra)
    # Test if charges are set correctly
    for spectrum in spectra_with_charge[:20]:
        assert spectrum.get("charge") == 1, "The charge is expected to be 1"
    for spectrum in spectra_with_charge[20:30]:
        assert spectrum.get("charge") == -1, "The charge is expected to be -1"
    for spectrum in spectra_with_charge[30:]:
        assert spectrum.get("charge") == 2, "The charge is expected to be 2"


def check_expected_headers(dataframe_found: pd.DataFrame,
                           expected_headers: List[str]):

    found_headers = list(dataframe_found.columns)
    assert len(found_headers) == len(found_headers)
    for i, header in enumerate(expected_headers):
        assert header == found_headers[i]