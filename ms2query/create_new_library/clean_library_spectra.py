from typing import List
from matchms import Spectrum
from matchmsextras.pubchem_lookup import pubchem_metadata_lookup
from tqdm import tqdm
import matchms.filtering as msfilters


def clean_up_smiles_inchi_and_inchikeys(spectra: List[Spectrum],
                                        do_pubchem_lookup) -> List[Spectrum]:
    """Uses filters to clean metadata of spectra

    do_pubchem_lookup: If true missing information will be searched on pubchem"""

    def run_metadata_filters(s):
        # Default filters
        s = msfilters.derive_adduct_from_name(s)
        s = msfilters.add_parent_mass(s, estimate_from_adduct=True)

        # Here, undefiend entries will be harmonized (instead of having a huge variation of None,"", "N/A" etc.)
        s = msfilters.harmonize_undefined_inchikey(s)
        s = msfilters.harmonize_undefined_inchi(s)
        s = msfilters.harmonize_undefined_smiles(s)

        # The repair_inchi_inchikey_smiles function will correct misplaced metadata
        # (e.g. inchikeys entered as inchi etc.) and harmonize the entry strings.
        s = msfilters.repair_inchi_inchikey_smiles(s)

        # Where possible (and necessary, i.e. missing): Convert between smiles, inchi, inchikey to complete metadata.
        # This is done using functions from rdkit.
        s = msfilters.derive_inchi_from_smiles(s)
        s = msfilters.derive_smiles_from_inchi(s)
        s = msfilters.derive_inchikey_from_inchi(s)

        if do_pubchem_lookup:
            s = pubchem_metadata_lookup(s,
                                        mass_tolerance=2.0,
                                        allowed_differences=[(18.03, 0.01),
                                                             (18.01, 0.01)],
                                        name_search_depth=15)
        return s

    return [run_metadata_filters(s) for s in tqdm(spectra,
                                                  desc="Cleaning metadata library spectra")]


def remove_wrong_ion_modes(spectra, ion_mode_to_keep):
    assert ion_mode_to_keep in {"positive", "negative"}, "ion_mode should be set to 'positive' or 'negative'"
    spectra_to_keep = []
    for spec in tqdm(spectra, desc=f"Selecting {ion_mode_to_keep} mode spectra"):
        if spec.get("ionmode") == ion_mode_to_keep:
            spectra_to_keep.append(spec)
    print(f"From {len(spectra)} spectra, "
          f"{len(spectra) - len(spectra_to_keep)} are removed since they are not in {ion_mode_to_keep} mode")
    return spectra_to_keep


def remove_not_fully_annotated_spectra(spectra: List[Spectrum]) -> List[Spectrum]:
    fully_annotated_spectra = []
    for spectrum in spectra:
        inchikey = spectrum.get("inchikey")
        if inchikey is not None and len(inchikey) > 13:
            smiles = spectrum.get("smiles")
            inchi = spectrum.get("inchi")
            if smiles is not None and len(smiles) > 0:
                if inchi is not None and len(inchi) > 0:
                    fully_annotated_spectra.append(spectrum)
    print(f"From {len(spectra)} spectra, "
          f"{len(spectra) - len(fully_annotated_spectra)} are removed since they are not fully annotated")
    return fully_annotated_spectra


def clean_peaks_and_normalise_intensities_spectra(spectra: List[Spectrum]) -> List[Spectrum]:
    """Cleans peaks of spectra

    pubchem_lookup:

    """

    def normalize_and_filter_peaks(spectrum):
        spectrum = msfilters.normalize_intensities(spectrum)
        spectrum = msfilters.select_by_relative_intensity(spectrum, 0.001, 1)
        spectrum = msfilters.select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)
        spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=500)
        spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=3)
        return spectrum

    spectra = [normalize_and_filter_peaks(s) for s in tqdm(spectra,
                                                           desc="Cleaning and filtering peaks library spectra")]
    spectra = [s for s in spectra if s is not None]
    return spectra
