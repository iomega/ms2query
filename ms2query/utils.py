import json
from matchms.importing.load_from_json import as_spectrum


def json_loader(file):
    """  

    Parameters
    ----------
    file
        Filename of json to import.
    """
    spectrums = json.load(file)
    if not isinstance(spectrums, list):
        spectrums = [spectrums]
    for i, spec_i in enumerate(spectrums):
        spectrum = as_spectrum(spec_i)
        if spectrum is not None:
            spectrums[i] = spectrum

    return spectrums
