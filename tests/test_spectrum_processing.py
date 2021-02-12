import numpy as np
from matchms import Spectrum
from ms2query.spectrum_processing import require_peaks_below_mz, \
    spectrum_processing_minimal, spectrum_processing_s2v



def test_require_peaks_below_mz_no_params():
    """Test default"""
    mz = np.array([110, 220, 330, 440], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_peaks_below_mz(spectrum_in)

    assert spectrum is None, "Expected None because the number of peaks (4) is less than the default threshold (10)."


def test_require_peaks_below_mz_required_4():
    """Test with adjustment of n_required."""
    mz = np.array([110, 220, 330, 440], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_peaks_below_mz(spectrum_in, n_required=4)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                "required number (4)."


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

    assert spectrum is None, "Expected different handling of None spectrum."


def test_spectrum_processing_minimal_default():
    spectrum_in = Spectrum(mz=np.array([110, 220, 330, 440], dtype="float"),
                           intensities=np.array([10, 1, 10, 100], dtype="float"))
    spectrum = spectrum_processing_minimal(spectrum_in)

    assert spectrum is None, \
        "Expected None because the number of peaks (4) is less than the default threshold (10)."


def test_spectrum_processing_minimal_set_n_required():
    spectrum_in = Spectrum(mz=np.array([5, 110, 220, 330, 399, 440], dtype="float"),
                           intensities=np.array([10, 10, 1, 10, 20, 100], dtype="float"))
    spectrum = spectrum_processing_minimal(spectrum_in, n_required_below_1000=4, max_mz_required=400)

    assert np.all(spectrum.peaks.mz == spectrum_in.peaks.mz[1:]), "Expected different m/z values"
    assert np.all(spectrum.peaks.intensities == np.array([0.1 , 0.01, 0.1 , 0.2 , 1.  ])), \
        "Expected different, normalized intensities"


def test_spectrum_processing_minimal_set_n_required_intensity_from():
    spectrum_in = Spectrum(mz=np.array([5, 110, 220, 330, 440], dtype="float"),
                           intensities=np.array([10, 10, 0.1, 10, 101], dtype="float"))
    spectrum = spectrum_processing_minimal(spectrum_in, n_required_below_1000=4)

    assert spectrum is None, \
        "Expected None because 1 peak has m/z > max_mz_required and 1 peak has intensity < intensity_from"


def test_spectrum_processing_s2v():
    """Test processing an individual spectrum with spectrum_processing_s2v"""
    spectrum_in = Spectrum(mz=np.array([5, 110, 220, 330, 440], dtype="float"),
                           intensities=np.array([0.1, 0.2, 0.1, 1, 0.5], dtype="float"),
                           metadata={"parent_mass": 250.0,
                                     "precursor_mz": 240.0})
    spectrum = spectrum_processing_s2v(spectrum_in)
    assert isinstance(spectrum, Spectrum), "Expected output to be Spectrum."
    assert spectrum.peaks == spectrum_in.peaks, "Expected spectrum peaks to be unchanged by function"
    assert np.all(spectrum.losses.mz == np.array([ 20., 130.])), "Expected other losses"


# TODO: uncomment once matchms n_max issue is fixed (#177 in matchms)
# def test_spectrum_processing_s2v_set_n_max():
#     """Test processing an individual spectrum with spectrum_processing_s2v"""
#     spectrum_in = Spectrum(mz=np.array([5, 110, 220, 330, 440], dtype="float"),
#                            intensities=np.array([0.1, 0.2, 0.1, 1, 0.5], dtype="float"),
#                            metadata={"parent_mass": 250.0,
#                                      "precursor_mz": 240.0})
#     spectrum = spectrum_processing_s2v(spectrum_in, n_max=4)
#     assert isinstance(spectrum, Spectrum), "Expected output to be Spectrum."
#     assert spectrum.peaks == spectrum_in.peaks[1:], "Expected spectrum peaks to be unchanged by function"
#     assert np.all(spectrum.losses.mz == np.array([ 20., 130.])), "Expected other losses"
