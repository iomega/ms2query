from matchms.Spectrum import Spectrum
from ms2query.s2v_functions import *
from ms2query.utils import json_loader
import os


def test_post_process_s2v():
    """Test processing an individual spectrum with post_process_s2v"""
    path_tests = os.path.dirname(__file__)
    testfile = os.path.join(path_tests, "testspectrum_query.json")
    spectrums = json_loader(open(testfile))
    spectrum = post_process_s2v(spectrums[0])
    assert isinstance(spectrum, Spectrum), "Expected output to be Spectrum."
    assert spectrums[0].metadata["spectrum_id"] == spectrum\
        .metadata["spectrum_id"]

def test_set_spec2vec_defaults():
    """"Test set_spec2vec_defaults"""
    x = 500
    settings = set_spec2vec_defaults(**{"mz_to": x})
    assert isinstance(settings, dict), "Expected output is dict"
    assert settings["mz_to"] == x
