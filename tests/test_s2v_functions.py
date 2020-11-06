from matchms.Spectrum import Spectrum
from ms2query.s2v_functions import post_process_s2v
from ms2query.utils import json_loader
import os


def test_post_process_s2v():
    """Test processing and individual spectrum with post_process_s2v"""
    testfile = os.path.join(path_tests, "testspectrum_query.json")
    spectrums = json_loader(open(testfile))
    spectrum = post_process_s2v(spectrums[0])
    assert isinstance(spectrum, Spectrum), "Expected output to be Spectrum."