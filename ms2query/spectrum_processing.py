"""Functions to handle spectrum processing steps. Any metadata related cleaning
and inspection is expected to happen prior to running MS2Query and is not taken
into account here. Processing here hence refers to inspecting, filtering,
adjusting the spectrum peaks (m/z and intensities).
"""
from typing import Dict, Union
import numpy as np
from matchms.typing import SpectrumType
from matchms.filtering import normalize_intensities, \
    select_by_mz, select_by_intensity, reduce_to_number_of_peaks, add_losses


def set_minimal_processing_defaults(**settings: Union[int, float]
                                    ) -> Dict[str, Union[int, float]]:
    """Set default argument values (where no user input is given).

    Args:
    ----------
    **settings:
        Change default settings
    """
    defaults = {"mz_from": 10.0,
                "n_required_below_mz": 5,
                "intensity_from": 0.001,
                "max_mz_required": 1000.0,
                }

    # Set default parameters or replace by **settings input
    for key in defaults:
        if key not in settings:
            settings[key] = defaults[key]
    return settings


def spectrum_processing_minimal(spectrum: SpectrumType,
                                **settings: Dict[str, Union[int, float]]
                                ) -> Union[SpectrumType, None]:
    """Minimal necessary spectrum processing that is required by MS2Query.
    This mostly includes intensity normalization and setting spectra to None
    when they do not meet the minimum requirements.

    Args:
    ----------
    spectrum:
        Spectrum to process
    mz_from
        Set lower threshold for m/z peak positions. Default is 10.0.
    n_required_below_mz
        Number of minimal required peaks with m/z below 1000.0Da for a spectrum
        to be considered.
        Spectra not meeting this criteria will be set to None.
    intensity_from
        Set lower threshold for peak intensity. Default is 0.001.
    max_mz_required
        Only peaks <= max_mz_required will be counted to check if spectrum
        contains sufficient peaks to be considered.
    """
    settings = set_minimal_processing_defaults(**settings)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_intensity(spectrum,
                                   intensity_from=settings["intensity_from"])
    spectrum = select_by_mz(spectrum,
                            mz_from=settings["mz_from"],
                            mz_to=np.inf)
    spectrum = require_peaks_below_mz(
        spectrum,
        n_required=settings["n_required_below_mz"],
        max_mz=settings["max_mz_required"])
    return spectrum


def set_spec2vec_defaults(**settings: Union[int, float]
                          ) -> Dict[str, Union[int, float]]:
    """Set spec2vec default argument values"(where no user input is given)".

    Args
    ----------
    **settings:
        Change default settings
    """
    defaults = {"mz_from": 10.0,
                "n_required": 1,
                "ratio_desired": 0.5,
                "intensity_from": 0.001,
                "n_max": 1000,
                "loss_mz_from": 5.0,
                "loss_mz_to": 200.0,
                }
    # Set default parameters or replace by **settings input
    for key in defaults:
        if key not in settings:
            settings[key] = defaults[key]
    return settings


def spectrum_processing_s2v(spectrum: SpectrumType,
                            **settings: Dict[str, Union[int, bool]]
                            ) -> Union[SpectrumType, None]:
    """Spectrum processing required for computing Spec2Vec scores.

    Args:
    ----------
    spectrum:
        Spectrum to process
    n_required
        Number of minimal required peaks for a spectrum to be considered.
    ratio_desired
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'. Default is 1.
    n_max
        Maximum number of peaks to be kept per spectrum. Default is 1000.
    loss_mz_from
        Minimum allowed m/z value for losses. Default is 0.0.
    loss_mz_to
        Maximum allowed m/z value for losses. Default is 1000.0.
    """
    settings = set_spec2vec_defaults(**settings)
    spectrum = reduce_to_number_of_peaks(
        spectrum,
        n_required=settings["n_required"],
        ratio_desired=settings["ratio_desired"],
        n_max=settings["n_max"])

    spectrum = add_losses(spectrum,
                          loss_mz_from=settings["loss_mz_from"],
                          loss_mz_to=settings["loss_mz_to"])
    return spectrum


def require_peaks_below_mz(spectrum_in: SpectrumType,
                           n_required: int = 10,
                           max_mz: float = 1000.0) -> SpectrumType:
    """Spectrum will be set to None when it has fewer peaks than required.

    Args:
    ----------
    spectrum_in:
        Input spectrum.
    n_required:
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'.
    max_mz:
        Only peaks <= max_mz will be counted to check if spectrum contains
        sufficient peaks to be considered (>= n_required).
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.peaks.mz[spectrum.peaks.mz < max_mz].size < n_required:
        return None

    return spectrum
