from typing import List, Tuple
import matchms.filtering as msfilters
from tqdm import tqdm
from matchms import Spectrum
from matchms.utils import is_valid_inchi, is_valid_inchikey, is_valid_smiles
from matchms.typing import SpectrumType
from matchms.logging_functions import set_matchms_logger_level
from matchmsextras.pubchem_lookup import pubchem_metadata_lookup
from spec2vec import SpectrumDocument


def clean_metadata(spectrum: Spectrum) -> Spectrum:
    spectrum = msfilters.default_filters(spectrum)
    spectrum = msfilters.add_retention_index(spectrum)
    spectrum = msfilters.add_retention_time(spectrum)
    spectrum = msfilters.require_precursor_mz(spectrum)
    return spectrum


def normalize_and_filter_peaks(spectrum: Spectrum) -> Spectrum:
    """Spectrum is normalized and filtered"""
    spectrum = msfilters.normalize_intensities(spectrum)
    spectrum = msfilters.select_by_relative_intensity(spectrum, 0.001, 1)
    spectrum = msfilters.select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)
    spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=500)
    spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=3)
    return spectrum


def create_spectrum_documents(query_spectra: List[Spectrum],
                              progress_bar: bool = False,
                              nr_of_decimals: int = 2
                              ) -> List[SpectrumDocument]:
    """Transforms list of Spectrum to List of SpectrumDocument

    Args
    ------
    query_spectra:
        List of Spectrum objects that are transformed to SpectrumDocument
    progress_bar:
        When true a progress bar is shown. Default = False
    nr_of_decimals:
        The number of decimals used for binning the peaks.
    """
    spectrum_documents = []
    for spectrum in tqdm(query_spectra,
                         desc="Converting Spectrum to Spectrum_document",
                         disable=not progress_bar):
        spectrum = msfilters.add_losses(spectrum,
                                        loss_mz_from=5.0,
                                        loss_mz_to=200.0)
        spectrum_documents.append(SpectrumDocument(
            spectrum,
            n_decimals=nr_of_decimals))
    return spectrum_documents


def harmonize_annotation(spectrum: Spectrum,
                         do_pubchem_lookup) -> Spectrum:
    set_matchms_logger_level("CRITICAL")
    # Here, undefiend entries will be harmonized (instead of having a huge variation of None,"", "N/A" etc.)
    spectrum = msfilters.harmonize_undefined_inchikey(spectrum)
    spectrum = msfilters.harmonize_undefined_inchi(spectrum)
    spectrum = msfilters.harmonize_undefined_smiles(spectrum)

    # The repair_inchi_inchikey_smiles function will correct misplaced metadata
    # (e.g. inchikeys entered as inchi etc.) and harmonize the entry strings.
    spectrum = msfilters.repair_inchi_inchikey_smiles(spectrum)

    # Where possible (and necessary, i.e. missing): Convert between smiles, inchi, inchikey to complete metadata.
    # This is done using functions from rdkit.
    spectrum = msfilters.derive_inchi_from_smiles(spectrum)
    spectrum = msfilters.derive_smiles_from_inchi(spectrum)
    spectrum = msfilters.derive_inchikey_from_inchi(spectrum)

    # Adding parent mass is relevant for pubchem lookup
    if do_pubchem_lookup:
        if not check_fully_annotated(spectrum):
            spectrum = msfilters.add_parent_mass(spectrum, estimate_from_adduct=True)
            spectrum = pubchem_metadata_lookup(spectrum,
                                               mass_tolerance=2.0,
                                               allowed_differences=[(18.03, 0.01),
                                                                    (18.01, 0.01)],
                                               name_search_depth=15)
    return spectrum


def remove_wrong_ion_modes(spectra, ion_mode_to_keep):
    assert ion_mode_to_keep in {"positive", "negative"}, "ion_mode should be set to 'positive' or 'negative'"
    spectra_to_keep = []
    for spectrum in tqdm(spectra,
                         desc=f"Selecting {ion_mode_to_keep} mode spectra"):
        if spectrum is not None:
            if spectrum.get("ionmode") == ion_mode_to_keep:
                spectra_to_keep.append(spectrum)
    print(f"From {len(spectra)} spectra, "
          f"{len(spectra) - len(spectra_to_keep)} are removed since they are not in {ion_mode_to_keep} mode")
    return spectra_to_keep


def check_fully_annotated(spectrum: Spectrum) -> bool:
    if not is_valid_smiles(spectrum.get("smiles")):
        return False
    if not is_valid_inchikey(spectrum.get("inchikey")):
        return False
    if not is_valid_inchi(spectrum.get("inchi")):
        return False
    spectrum = msfilters.require_precursor_mz(spectrum)
    if spectrum is None:
        return False
    return True


def split_annotated_spectra(spectra: List[Spectrum]) -> Tuple[List[Spectrum], List[Spectrum]]:
    fully_annotated_spectra = []
    not_fully_annotated_spectra = []
    for spectrum in tqdm(spectra,
                         desc="Splitting annotated and unannotated spectra"):
        fully_annotated = check_fully_annotated(spectrum)
        if fully_annotated:
            fully_annotated_spectra.append(spectrum)
        else:
            not_fully_annotated_spectra.append(spectrum)
    print(f"From {len(spectra)} spectra, "
          f"{len(spectra) - len(fully_annotated_spectra)} are removed since they are not fully annotated")
    return fully_annotated_spectra, not_fully_annotated_spectra


def normalize_and_filter_peaks_multiple_spectra(spectrum_list: List[SpectrumType],
                                                progress_bar: bool = False
                                                ) -> List[SpectrumType]:
    """Preprocesses all spectra and removes None values

    Args:
    ------
    spectrum_list:
        List of spectra that should be preprocessed.
    progress_bar:
        If true a progress bar will be shown.
    """
    for i, spectrum in enumerate(
            tqdm(spectrum_list,
                 desc="Preprocessing spectra",
                 disable=not progress_bar)):
        processed_spectrum = normalize_and_filter_peaks(spectrum)
        spectrum_list[i] = processed_spectrum

    # Remove None values
    return [spectrum for spectrum in spectrum_list if spectrum]


def clean_normalize_and_split_annotated_spectra(spectra: List[Spectrum],
                                                ion_mode_to_keep,
                                                do_pubchem_lookup=True) -> Tuple[List[Spectrum], List[Spectrum]]:
    spectra = [clean_metadata(s) for s in tqdm(spectra, desc="Cleaning metadata")]
    spectra = remove_wrong_ion_modes(spectra, ion_mode_to_keep)
    spectra = [harmonize_annotation(s, do_pubchem_lookup) for s in tqdm(spectra, desc="Harmonizing annotations")]
    spectra = normalize_and_filter_peaks_multiple_spectra(spectra, progress_bar=True)
    annotated_spectra, unannotated_spectra = split_annotated_spectra(spectra)
    # Both annotated and unannotated spectra are returned to make it possible to still use them for Spec2Vec training
    return annotated_spectra, unannotated_spectra
