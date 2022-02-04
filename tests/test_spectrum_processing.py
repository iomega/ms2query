import os
import sys
import numpy as np
from matchms import Spectrum
from spec2vec import SpectrumDocument
from ms2query.spectrum_processing import (clean_metadata,
                                          create_spectrum_documents,
                                          minimal_processing_multiple_spectra,
                                          require_peaks_below_mz,
                                          spectrum_processing_minimal,
                                          spectrum_processing_s2v)


if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


def test_minimal_processing_multiple_spectra():
    spectrum_1 = Spectrum(mz=np.array([5, 110, 220, 330, 399, 440],
                                       dtype="float"),
                          intensities=np.array([10, 10, 1, 10, 20, 100],
                                               dtype="float"),
                          metadata={"precursor_mz": 240.0})

    spectrum_2 = Spectrum(mz=np.array([110, 220, 330], dtype="float"),
                          intensities=np.array([0, 1, 10], dtype="float"),
                          metadata={"precursor_mz": 240.0}
                          )
    spectrum_list = [spectrum_1, spectrum_2]
    processed_spectrum_list = minimal_processing_multiple_spectra(
        spectrum_list,
        n_required_below_mz=4,
        max_mz_required=400)
    assert len(processed_spectrum_list) == 1, \
        "Expected only 1 spectrum, since spectrum 2 does not have enough peaks"
    found_spectrum = processed_spectrum_list[0]
    assert np.all(found_spectrum.peaks.mz == spectrum_1.peaks.mz[1:]), \
        "Expected different m/z values"
    assert np.all(found_spectrum.peaks.intensities ==
                  np.array([0.1, 0.01, 0.1, 0.2, 1.])),\
        "Expected different intensities"


def test_require_peaks_below_mz_no_params():
    """Test default"""
    mz = np.array([110, 220, 330, 440], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_peaks_below_mz(spectrum_in)

    assert spectrum is None, \
        "Expected None, because the number of peaks (4) is less than the " \
        "default threshold (10)."


def test_require_peaks_below_mz_required_4():
    """Test with adjustment of n_required."""
    mz = np.array([110, 220, 330, 440], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_peaks_below_mz(spectrum_in, n_required=4)

    assert spectrum == spectrum_in, \
        "Expected the spectrum to qualify because the number of peaks (4) " \
        "is equal to the required number (4)."


def test_require_peaks_below_mz_required_4_below_max_mz():
    """Test with adjustment of n_required but with lower max_mz."""
    mz = np.array([110, 220, 330, 440], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_peaks_below_mz(spectrum_in, n_required=4, max_mz=400)

    assert spectrum is None, \
        "Expected None since peaks with mz<=max_mz is less than n_required."


def test_require_peaks_below_empty_spectrum():
    spectrum_in = None
    spectrum = require_peaks_below_mz(spectrum_in)

    assert spectrum is None, \
        "Expected different handling of None spectrum."


def test_spectrum_processing_minimal_default():
    spectrum_in = Spectrum(mz=np.array([110, 220, 330, 440], dtype="float"),
                           intensities=np.array([10, 1, 10, 100],
                                                dtype="float"))
    spectrum = spectrum_processing_minimal(spectrum_in)

    assert spectrum is None, \
        "Expected None because the number of peaks (4) is less than the " \
        "default threshold (10)."


def test_spectrum_processing_minimal_set_n_required_and_max_mz():
    spectrum_in = Spectrum(mz=np.array([5, 110, 220, 330, 399, 440],
                                       dtype="float"),
                           intensities=np.array([10, 10, 1, 10, 20, 100],
                                                dtype="float"),
                           metadata={"precursor_mz": 240.0}
                           )
    spectrum = spectrum_processing_minimal(spectrum_in, n_required_below_mz=4,
                                           max_mz_required=400)

    assert np.all(spectrum.peaks.mz == spectrum_in.peaks.mz[1:]), \
        "Expected different m/z values"
    assert np.all(spectrum.peaks.intensities ==
                  np.array([0.1, 0.01, 0.1, 0.2, 1.])),\
        "Expected different intensities"


def test_spectrum_processing_minimal_set_n_required_intensity_from():
    spectrum_in = Spectrum(mz=np.array([5, 110, 220, 330, 440],
                                       dtype="float"),
                           intensities=np.array([10, 10, 0.1, 10, 101],
                                                dtype="float"))
    spectrum = spectrum_processing_minimal(spectrum_in,
                                           n_required_below_mz=4)

    assert spectrum is None, \
        "Expected None because 1 peak has m/z > max_mz_required and 1 peak " \
        "has intensity < intensity_from"


def test_spectrum_processing_s2v():
    """Test processing an individual spectrum with spectrum_processing_s2v"""
    spectrum_in = Spectrum(mz=np.array([110, 220, 330, 440, 1050],
                                       dtype="float"),
                           intensities=np.array([0.1, 0.2, 0.1, 1, 0.5],
                                                dtype="float"),
                           metadata={"precursor_mz": 240.0})
    spectrum = spectrum_processing_s2v(spectrum_in)
    assert isinstance(spectrum, Spectrum), "Expected output to be Spectrum."
    assert np.all(spectrum.peaks.mz == spectrum_in.peaks.mz[:-1]), \
        "Expected the last peak to be removed, the rest should be unchanged"
    assert np.all(spectrum.peaks.intensities ==
                  spectrum_in.peaks.intensities[:-1]),         \
        "Expected the last peak to be removed, the rest should be unchanged"
    assert np.all(spectrum.losses.mz == np.array([20., 130.])), \
        "Expected other losses"


# TODO: uncomment once matchms n_max issue is fixed (#177 in matchms)
# def test_spectrum_processing_s2v_set_n_max():
#     """Test processing an individual spectrum with spectrum_processing_s2v"""
#     spectrum_in = Spectrum(mz=np.array([5, 110, 220, 330, 440],
#                                        dtype="float"),
#                            intensities=np.array([0.1, 0.2, 0.1, 1, 0.5],
#                                                 dtype="float"),
#                            metadata={"precursor_mz": 250.0})
#     spectrum = spectrum_processing_s2v(spectrum_in, n_max=4)
#     assert isinstance(spectrum, Spectrum), "Expected output to be Spectrum."
#     assert spectrum.peaks == spectrum_in.peaks[1:], \
#         "Expected only the first peak to be removed"
#     assert np.all(spectrum.losses.mz == np.array([ 20., 130.])), \
#         "Expected other losses"
def test_create_spectrum_documents():
    path_to_pickled_file = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/first_10_spectra.pickle')
    with open(path_to_pickled_file, "rb") as pickled_file:
        spectrum_list = pickle.load(pickled_file)
    spectrum_list = minimal_processing_multiple_spectra(spectrum_list)

    spectrum_documents = create_spectrum_documents(spectrum_list)
    assert isinstance(spectrum_documents, list), \
        "A list with spectrum_documents is expected"
    for spectrum_doc in spectrum_documents:
        assert isinstance(spectrum_doc, SpectrumDocument), \
            "A list with spectrum_documents is expected"
