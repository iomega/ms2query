"""Functions to handle spectrum processing steps. Any metadata related cleaning
and inspection is expected to happen prior to running MS2Query and is not taken
into account here. Processing here hence refers to inspecting, filtering, adjusting
the spectrum peaks (m/z and intensities).
"""
from matchms.typing import SpectrumType

def set_minimal_processing_defaults(**settings: Dict[str, Union[int, float]]) \
        -> Dict[str, Union[int, bool]]:
    """Set default argument values (where no user input is given).

    Args:
    ----------
    **settings:
        Change default settings
    """
    defaults = {"mz_from": 10.0,
                "n_required_below_1000": 5,
                "intensity_from": 0.001,
                "max_mz_required": 1000.0,
                }

    # Set default parameters or replace by **settings input
    for key in defaults:
        if key not in settings:
            settings[key] = defaults[key]
    return settings


def spectrum_processing_minimal(spectrums: SpectrumType,
                                **settings: Dict[str, Union[int, float]) -> Union[Spectrum, None]:
    """Minimal necessary spectrum processing that is required by MS2Query.
    This mostly includes intensity normalization and setting spectrums to None
    when they do not meet the minimum requirements.

    Args:
    ----------
    spectrum:
        Spectrum to process
    mz_from
        Set lower threshold for m/z peak positions. Default is 0.0.
    n_required_below_1000
        Number of minimal required peaks with m/z below 1000.0Da for a spectrum
        to be considered. Spectrums not meeting this criteria will be set to None.
    intensity_from
        Set lower threshold for peak intensity. Default is 10.0.
    max_mz_required
        Only peaks <= max_mz_required will be counted to check if spectrum
        contains sufficient peaks to be considered.
    """
    settings = set_minimal_processing_defaults(**settings)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=settings["mz_from"], mz_to=None)
    spectrum = require_peaks_below_mz(spectrum, n_required=settings["n_required_below_1000"]
                                      max_mz=settings["max_mz_required"])
    return spectrum


def require_peaks_below_mz(spectrum_in: SpectrumType,
                           n_required: int = 10,
                           max_mz: float = 1000.0) -> SpectrumType:
    """Spectrum will be set to None when it has fewer peaks than required.

    Parameters
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
