import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from spec2vec import SpectrumDocument
from tensorflow.keras.models import load_model


# pylint: disable=protected-access
def find_info_matches(matches: List[pd.DataFrame],
                      documents_library: List[SpectrumDocument],
                      documents_query: List[SpectrumDocument],
                      add_cols: List[str] = None,
                      add_num_matches_transform: bool = True,
                      add_mass_transform: bool = False,
                      max_parent_mass: float = None,
                      add_mass_similarity: bool = True):
    """
    To each df in matches, add/alter info like similarity of parent masses

    Existing values can be transformed between 0-1. List of matches is returned
    in the same order but with altered/new columns

    Args:
    -------
    matches:
        Library matching result of query on library
    documents_library:
        Spectra in library
    documents_query:
        Spectra in query set. Indices should correspond to indices of matches.
    add_cols:
        Add other info present in metadata such as parent_mass, adduct
    add_num_matches_transform:
        Transform cosine and mod_cosine matches to a number 0-1. Both matches
        are transformed to between 0-1 by doing 1-0.93^num_matches
    add_mass_transform:
        Add transform of the parent masses to a fraction of the maximal parent
        mass
    max_parent_mass:
        The maximum parent mass in the dataset
    add_mass_similarity:
        Add similarity of parent mass to the query as a scaled number from 0-1
        where The similarity in dalton is calculated and transformed into a
        value 0-1 by doing 1 - base_num^diff_in_dalton
    """
    # pylint: disable=too-many-arguments
    matches_with_info = []
    if add_mass_transform and not max_parent_mass:
        print("""If you want to transform the masses, please provide a
        max_parent_mass""")
        return None
    if max_parent_mass:
        print('Max parent mass:', max_parent_mass)

    for query_id, document_query in enumerate(documents_query):
        match = matches[query_id].copy()
        if add_cols:
            match = find_basic_info(match, documents_library, add_cols)
        if add_num_matches_transform:
            match = transform_num_matches(match)
        if add_mass_transform:
            # add parent_mass if its not there already
            match = find_basic_info(match, documents_library,
                                    add_cols=['parent_mass'])
            match['parent_mass'] = [cur_pm / max_parent_mass for cur_pm in
                                    match['parent_mass']]
        if add_mass_similarity:
            if 'mass_match' in match:
                match.drop(['mass_match'], axis=1, inplace=True)
            q_mass = document_query._obj.get("parent_mass")
            match = find_mass_similarity(match, documents_library, q_mass,
                                         base_num=0.8)

        matches_with_info.append(match)
    return matches_with_info


def find_basic_info(matches: pd.DataFrame,
                    documents_library: List[SpectrumDocument],
                    add_cols: Union[List[str], Tuple[str]] = ('parent_mass',)):
    """
    To each match in matches df, add the info from add_cols entries

    Args:
    -------
    matches:
        Library matching result of 1 query on library
    documents_library:
        Spectra in library
    add_cols:
        List of the metadata categories to add as columns
    """
    df = matches.copy()
    library_ids = df.index.values
    if add_cols:
        for col in add_cols:
            col_data = []
            for lib_id in library_ids:
                lib_data = documents_library[lib_id]._obj.get(col)
                col_data.append(lib_data)
            df[col] = col_data
    return df


def transform_num_matches(matches: pd.DataFrame):
    """Transform the cosine_matches and mod_cosine_matches to between 0-1

    Args:
    -------
    matches:
        Library matching result of 1 query on library, must contain
        cosine_matches and mod_cosinge_matches columns

    Both matches are transformed to between 0-1 by doing 1-0.93^num_matches
    """
    df = matches.copy()  # otherwise it edits the df outside the function
    df['cosine_matches'] = [(1 - 0.93 ** i) for i in df['cosine_matches']]
    df['mod_cosine_matches'] = [(1 - 0.93 ** i) for i in
                                df['mod_cosine_matches']]
    return df


def find_mass_similarity(matches: pd.DataFrame,
                         documents_library: List[SpectrumDocument],
                         query_mass: float,
                         base_num: float = 0.8):
    """
    Add scaled value 0-1 mass_sim of how similar match parent mass is to query

    Args:
    -------
    matches:
        Library matching result of 1 query on library
    documents_library:
        Spectra in library
    query_mass:
        Parent mass of query
    base_num:
        The base for the exponent

    The similarity in dalton is calculated and transformed into a value 0 - 1
    by doing 1 - base_num^diff_in_dalton
    """
    df = matches.copy()
    library_ids = df.index.values
    scaled_mass_sims = []
    for lib_id in library_ids:
        lib_mass = documents_library[lib_id]._obj.get("parent_mass")
        mass_diff = abs(lib_mass - query_mass)
        scaled_mass_sim = base_num ** mass_diff
        scaled_mass_sims.append(scaled_mass_sim)

    # add to df
    df['mass_sim'] = scaled_mass_sims
    return df


def nn_predict_on_matches(matches: pd.DataFrame,
                          documents_library: List[SpectrumDocument],
                          documents_query: List[SpectrumDocument],
                          model_path: str,
                          max_parent_mass: float) -> np.ndarray:
    """Returns predicted tanimoto scores from the model in order of lib matches

    Args:
    ------
    matches:
        Library matching result of 1 query on library
    documents_library:
        Spectra in library
    documents_query:
        Spectra in query set. Indices should correspond to indices of matches.
    model_path:
        Path to neural network
    max_parent_mass:
        The maximum parent mass used in the model, read from model_metadata
        file
    """
    model = load_model(model_path)
    prepped_matches = find_info_matches(
        [matches], documents_library, documents_query, add_mass_transform=True,
        max_parent_mass=max_parent_mass)
    predictions = model.predict(prepped_matches)
    return predictions
