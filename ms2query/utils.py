from matchms.importing.load_from_json import as_spectrum


def json_loader(file: str):
    """
    

    Parameters
    ----------
    file
        Filename of json to import.
    """
    spectrums = json.load(file)
    if not isinstance(spectrums, list):
        spectrums = [spectrums]
    for i in range(len(spectrums)):
        spectrum = as_spectrum(spectrums[i])
        if spectrum is not None:
            spectrums[i] = spectrum

    return spectrums
