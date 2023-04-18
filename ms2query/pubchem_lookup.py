import logging
import re
import time
import pubchempy as pcp
import numpy as np
from matchms.utils import is_valid_inchikey


logger = logging.getLogger("matchms")


def pubchem_metadata_lookup(spectrum_in, name_search_depth=10,
                            match_precursor_mz=False,
                            formula_search=False,
                            mass_tolerance=2.0,
                            allowed_differences=((18.03, 0.01)),
                            min_formula_length=6,
                            formula_search_depth=25,
                            pause_per_request=0,
                            verbose=2):
    """
    Parameters
    ----------
    spectrum_in
        Matchms type spectrum as input.
    name_search_depth: int
        How many of the most relevant name matches to explore deeper. Default = 10.
    """
    if spectrum_in is None:
        return None

    # Only run search if no valid-looking inchikey is found
    if is_valid_inchikey(spectrum_in.get("inchikey")):
        return spectrum_in

    spectrum = spectrum_in.clone()

    def _plausible_name(compound_name):
        return (isinstance(compound_name, str) and len(compound_name) > 4)

    # Only run search if (more or less) plausible name is found
    compound_name = spectrum.get("compound_name")
    if not _plausible_name(compound_name):
        logger.info("No plausible compound name found (%s)", compound_name)
        return spectrum

    # Start pubchem search
    time.sleep(pause_per_request)
    inchi = spectrum.get("inchi")
    parent_mass = spectrum.get("parent_mass")
    if isinstance(parent_mass, np.ndarray):
        parent_mass = parent_mass[0]
    formula = spectrum.get("formula")

    # 1) Search for matching compound name
    results_pubchem = pubchem_name_search(compound_name, name_search_depth=name_search_depth)

    if len(results_pubchem) > 0:
        logger.info("Found potential matches for compound name (%s) on PubChem",
                   compound_name)

        # 1a) Search for matching inchi
        if likely_has_inchi(inchi):
            inchi_pubchem, inchikey_pubchem, smiles_pubchem = find_pubchem_inchi_match(results_pubchem, inchi,
                                                                                       verbose=verbose)
        # 1b) Search for matching parent mass
        if not likely_has_inchi(inchi) or inchikey_pubchem is None:
            inchi_pubchem, inchikey_pubchem, smiles_pubchem = find_pubchem_mass_match(results_pubchem,
                                                                                      parent_mass,
                                                                                      given_mass="parent mass",
                                                                                      mass_tolerance=mass_tolerance,
                                                                                      allowed_differences=allowed_differences)

        # 1c) Search for matching precursor mass (optional)
        if match_precursor_mz and inchikey_pubchem is None:
            precursor_mz = spectrum.get("precursor_mz")
            inchi_pubchem, inchikey_pubchem, smiles_pubchem = find_pubchem_mass_match(results_pubchem,
                                                                                      precursor_mz,
                                                                                      given_mass="precursor mass",
                                                                                      mass_tolerance=mass_tolerance,
                                                                                      allowed_differences=allowed_differences)

        if inchikey_pubchem is not None and inchi_pubchem is not None:
            logger.info("Matching compound name: %s", compound_name)
            spectrum.set("inchikey", inchikey_pubchem)
            spectrum.set("inchi", inchi_pubchem)
            spectrum.set("smiles", smiles_pubchem)
            return spectrum

        if verbose >= 2:
            logger.info("No matches found for compound name: %s", compound_name)

    else:
        logger.info("No matches for compound name (%s) on PubChem",
                   compound_name)

    # 2) Search for matching formula
    if formula_search and formula and len(formula) >= min_formula_length:
        results_pubchem = pubchem_formula_search(formula, formula_search_depth=formula_search_depth)

        if len(results_pubchem) > 0:
            inchikey_pubchem = None
            logger.info("Found potential matches for formula (%s) on PubChem",
                       formula)
            # 2a) Search for matching inchi
            if likely_has_inchi(inchi):
                inchi_pubchem, inchikey_pubchem, smiles_pubchem = find_pubchem_inchi_match(results_pubchem, inchi,
                                                                                           verbose=verbose)
            # 2b) Search for matching parent mass
            if inchikey_pubchem is None:
                inchi_pubchem, inchikey_pubchem, smiles_pubchem = find_pubchem_mass_match(results_pubchem,
                                                                                          parent_mass,
                                                                                          given_mass="parent mass",
                                                                                          mass_tolerance=mass_tolerance,
                                                                                          allowed_differences=allowed_differences)
            # 2c) Search for matching precursor mass (optional)
            if match_precursor_mz and inchikey_pubchem is None:
                precursor_mz = spectrum.get("precursor_mz")
                inchi_pubchem, inchikey_pubchem, smiles_pubchem = find_pubchem_mass_match(results_pubchem,
                                                                                          precursor_mz,
                                                                                          given_mass="precursor mass",
                                                                                          mass_tolerance=mass_tolerance,
                                                                                          allowed_differences=allowed_differences)
            if inchikey_pubchem is not None and inchi_pubchem is not None:
                logger.info("Matching formula: %s", formula)
                if verbose >= 1:
                    logger.info("Matching formula: %s", formula)
                spectrum.set("inchikey", inchikey_pubchem)
                spectrum.set("inchi", inchi_pubchem)
                spectrum.set("smiles", smiles_pubchem)
                return spectrum

            if verbose >= 2:
                logger.info("No matches found for formula: %s", formula)
        else:
            logger.info("No matches for formula (%s) on PubChem",
                        formula)

    return spectrum


def likely_has_inchi(inchi):
    """Quick test to avoid excess in-depth testing"""
    if inchi is None:
        return False
    inchi = inchi.strip('"')
    regexp = r"(InChI=1|1)(S\/|\/)[0-9, A-Z, a-z,\.]{2,}\/(c|h)[0-9]"
    if not re.search(regexp, inchi):
        return False
    return True


def likely_inchi_match(inchi_1, inchi_2, min_agreement=3):
    """Try to match defective inchi to non-defective ones.
    Compares inchi parts seperately. Match is found if at least the first
    'min_agreement' parts are a good enough match.
    The main 'defects' this method accounts for are missing '-' in the inchi.
    In addition, differences between '-', '+', and '?'will be ignored.
    Parameters
    ----------
    inchi_1: str
        inchi of molecule.
    inchi_2: str
        inchi of molecule.
    min_agreement: int
        Minimum number of first parts that MUST be a match between both input
        inchi to finally consider it a match. Default is min_agreement=3.
    """
    if min_agreement < 2:
        logger.warning("Warning! 'min_agreement' < 2 has no discriminative power. Should be => 2.")
    if min_agreement == 2:
        logger.warning("Warning! 'min_agreement' == 2 has little discriminative power",
                       "(only looking at structure formula. Better use > 2.")
    agreement = 0

    # Remove spaces and '"' to account for different notations.
    # Remove everything with little discriminative power.
    ignore_lst = ['"', ' ', '-', '+', '?']
    for ignore in ignore_lst:
        inchi_1 = inchi_1.replace(ignore, '')
        inchi_2 = inchi_2.replace(ignore, '')

    # Split inchi in parts.
    inchi_1_parts = inchi_1.split('/')
    inchi_2_parts = inchi_2.split('/')

    # Check if both inchi have sufficient parts (seperated by '/')
    if len(inchi_1_parts) >= min_agreement and len(
            inchi_2_parts) >= min_agreement:
        # Count how many parts agree well
        for i in range(min_agreement):
            agreement += (inchi_1_parts[i] == inchi_2_parts[i])

    return bool(agreement == min_agreement)


def pubchem_name_search(compound_name: str, name_search_depth=10):
    """Search pubmed for compound name"""
    results_pubchem = pcp.get_compounds(compound_name,
                                        'name',
                                        listkey_count=name_search_depth)
    if len(results_pubchem) == 0 and "_" in compound_name:
        results_pubchem = pcp.get_compounds(compound_name.replace("_", " "),
                                            'name',
                                            listkey_count=name_search_depth)
    if len(results_pubchem) == 0:
        return []

    logger.debug("Found at least %s compounds of that name on pubchem.", len(results_pubchem))
    return results_pubchem


def pubchem_formula_search(compound_formula: str, formula_search_depth=25):
    """Search pubmed for compound formula"""
    sids_pubchem = pcp.get_sids(compound_formula,
                                'formula',
                                listkey_count=formula_search_depth)

    results_pubchem = []
    for sid in sids_pubchem:
        result = pcp.Compound.from_cid(sid['CID'])
        results_pubchem.append(result)

    logger.debug("Found at least %s compounds of with formula: %s.",
                 len(results_pubchem), compound_formula)
    return results_pubchem


def find_pubchem_inchi_match(results_pubchem,
                             inchi,
                             min_inchi_match=3,
                             verbose=1):
    """Searches pubmed matches for inchi match.
    Then check if inchi can be matched to (defective) input inchi.
    Outputs found inchi and found inchikey (will be None if none is found).
    Parameters
    ----------
    results_pubchem: List[dict]
        List of name search results from Pubchem.
    inchi: str
        Inchi (correct, or defective...). Set to None to ignore.
    min_inchi_match: int
        Minimum number of first parts that MUST be a match between both input
        inchi to finally consider it a match. Default is min_inchi_match=3.
    """

    inchi_pubchem = None
    inchikey_pubchem = None
    smiles_pubchem = None

    # Loop through first 'name_search_depth' results found on pubchem. Stop once first match is found.
    for result in results_pubchem:
        inchi_pubchem = '"' + result.inchi + '"'
        inchikey_pubchem = result.inchikey
        smiles_pubchem = result.isomeric_smiles
        if smiles_pubchem is None:
            smiles_pubchem = result.canonical_smiles

        match_inchi = likely_inchi_match(inchi, inchi_pubchem,
                                         min_agreement=min_inchi_match)

        if match_inchi:
            logger.info("Matching inchi: %s", inchi)
            if verbose >= 1:
                logger.info("Found matching compound for inchi: %s (Pubchem: %s)",
                            inchi, inchi_pubchem)
            break

    if not match_inchi:
        inchi_pubchem = None
        inchikey_pubchem = None
        smiles_pubchem = None

        if verbose >= 2:
            logger.info("No matches found for inchi %s.", inchi)

    return inchi_pubchem, inchikey_pubchem, smiles_pubchem


def find_pubchem_mass_match(results_pubchem,
                            parent_mass,
                            mass_tolerance,
                            given_mass="parent mass",
                            allowed_differences=((18.03, 0.01))):
    """Searches pubmed matches for inchi match.
    Then check if inchi can be matched to (defective) input inchi.
    Outputs found inchi and found inchikey (will be None if none is found).
    Parameters
    ----------
    results_pubchem: List[dict]
        List of name search results from Pubchem.
    parent_mass: float
        Spectrum"s guessed parent mass.
    mass_tolerance: float
        Acceptable mass difference between query compound and pubchem result.
    given_mass
        String to specify the type of the given mass (e.g. "parent mass").
    """
    inchi_pubchem = None
    inchikey_pubchem = None
    smiles_pubchem = None
    lowest_mass_difference = [np.inf, None]

    for result in results_pubchem:
        inchi_pubchem = '"' + result.inchi + '"'
        inchikey_pubchem = result.inchikey
        smiles_pubchem = result.isomeric_smiles
        if smiles_pubchem is None:
            smiles_pubchem = result.canonical_smiles

        pubchem_mass = float(results_pubchem[0].exact_mass)
        mass_difference = np.abs(pubchem_mass - parent_mass)
        if mass_difference < lowest_mass_difference[0]:
            lowest_mass_difference[0] = mass_difference
            lowest_mass_difference[1] = inchi_pubchem
        match_mass = (mass_difference <= mass_tolerance)
        for diff in allowed_differences:
            match_mass = match_mass or np.isclose(mass_difference, diff[0], atol=diff[1])

        if match_mass:
            logger.info("Matching molecular weight (%s vs %s of %s)",
                        pubchem_mass, given_mass, parent_mass)
            break

    if not match_mass:
        inchi_pubchem = None
        inchikey_pubchem = None
        smiles_pubchem = None

        logger.info("No matching molecular weight (best mass difference was %s for inchi: %s)",
                    lowest_mass_difference[0], lowest_mass_difference[1])

    return inchi_pubchem, inchikey_pubchem, smiles_pubchem