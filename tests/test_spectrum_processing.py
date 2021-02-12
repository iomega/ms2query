import numpy as np
from matchms import Spectrum
from ms2query.spectrum_processing import require_peaks_below_mz, spectrum_processing_minimal


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


def test_empty_spectrum():
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
    spectrum_in = Spectrum(mz=np.array([5, 110, 220, 330, 440], dtype="float"),
                           intensities=np.array([10, 10, 1, 10, 100], dtype="float"))
    spectrum = spectrum_processing_minimal(spectrum_in, n_required_below_1000=4)

    assert np.all(spectrum.peaks.mz == spectrum_in.peaks.mz[1:]), "Expected m/z values to be unchanged"
    assert np.all(spectrum.peaks.intensities == np.array([0.1 , 0.01, 0.1 , 1.  ])), \
        "Expected different, normalized intensities"
