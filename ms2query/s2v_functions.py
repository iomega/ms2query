from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import add_losses


def set_spec2vec_defaults(**settings):
    """Set spec2vec default argument values"(where no user input is given)".

    Args
    ----------
    **settings: optional dict
        Change default settings
    """
    defaults = {"mz_from": 0,
                "mz_to": 1000,
                "n_required": 10,
                "ratio_desired": 0.5,
                "intensity_from": 0.001,
                "loss_mz_from": 5.0,
                "loss_mz_to": 200.0}

    # Set default parameters or replace by **settings input
    for key in defaults:
        if key not in settings:
            settings[key] = defaults[key]
    return settings


def post_process_s2v(spectrum, **settings):
    """Returns a processed matchms.Spectrum.Spectrum

    Args:
    ----------
    spectrum: matchms.Spectrum.Spectrum
        Spectrum to process
    **settings: optional dict
        Change default settings
    """
    settings = set_spec2vec_defaults(**settings)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=settings["mz_from"],
                            mz_to=settings["mz_to"])
    spectrum = require_minimum_number_of_peaks(spectrum,
                                               n_required=settings[
                                                   "n_required"])
    spectrum = reduce_to_number_of_peaks(spectrum,
                                         n_required=settings["n_required"],
                                         ratio_desired=settings
                                         ["ratio_desired"])
    if spectrum is None:
        return None
    s_remove_low_peaks = select_by_relative_intensity(spectrum,
                                                      intensity_from=settings
                                                      ["intensity_from"])
    if len(s_remove_low_peaks.peaks) >= 10:
        spectrum = s_remove_low_peaks

    spectrum = add_losses(spectrum, loss_mz_from=settings["loss_mz_from"],
                          loss_mz_to=settings["loss_mz_to"])
    return spectrum
