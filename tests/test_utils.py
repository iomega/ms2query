import os
from ms2query.utils import json_loader


def test_json_loader_testfile():
    """Test creating spectrum(s) from test json file."""
    testfile = os.path.join(os.getcwd(), "testspectrum.json")
    spectrums = json_loader(testfile)
    assert isinstance(spectrums, list), "Expected output to be list."
    assert spectrums[0].metadata["spectrum_id"] == "CCMSLIB00000479320"
    
