from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import add_losses


def post_process_s2v(spectrum,
                     mz_from=0,
                     mz_to=1000,
                     n_required=10,
                     ratio_desired=0.5,
                     intensity_from=0.001,
                     loss_mz_from=5.0,
                     loss_mz_to=200.0):
    """Process a spectrum to use it with spec2vec, returns SpectrumDocument

    Args:
    ----------
    spectrum: SpectrumDocument
        Spectrum to process
    mz_from: float, optional
        Set lower threshold for m/z peak positions. Default: 0
    mz_to: float, optional
        Set lower threshold for m/z peak positions. Default: 1000
    n_required: int, optional
        Number of minimum required peaks. Default: 10
    ratio_desired: float, optional
        Set desired ratio between maximum number of peaks and parent mass.
        Default: 0.5
    intensity_from: float, optional
        Set lower threshold for relative peak intensity. Default: 0.001
    loss_mz_from: float, optional
        Minimum allowed m/z value for losses. Default: 5.0
    loss_mz_to: float, optional
        Maximum allowed m/z value for losses. Default: 200.0
    """
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=mz_from, mz_to=mz_to)
    spectrum = require_minimum_number_of_peaks(spectrum,
                                               n_required=n_required)
    spectrum = reduce_to_number_of_peaks(spectrum, n_required=n_required,
                                         ratio_desired=ratio_desired)
    if spectrum is None:
        return None
    s_remove_low_peaks = select_by_relative_intensity(spectrum,
                                                intensity_from=intensity_from)
    if len(s_remove_low_peaks.peaks) >= 10:
        spectrum = s_remove_low_peaks

    spectrum = add_losses(spectrum, loss_mz_from=loss_mz_from,
                          loss_mz_to=loss_mz_to)
    return spectrum
