import pandas as pd
from typing import Union, List, Tuple
from spec2vec import SpectrumDocument


# pylint: disable=protected-access
def find_info_matches(matches: List[pd.DataFrame],
                      documents_library: List[SpectrumDocument],
                      documents_query: List[SpectrumDocument],
                      add_cols: List[str] = None,
                      add_num_matches_transform: bool = True,
                      add_mass_transform: bool = False,
                      max_parent_mass: float = None,
                      add_mass_similarity=True):
    """
    To each df in matches, add/alter info like similarity of parent masses

    Existing values can be transformed between 0-1. List of matches is returned
    in the same order but with altered/new columns

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
    matches_with_info = []
    if add_mass_transform and not max_parent_mass:
        print("""If you want to transform the masses, please provide a
        max_parent_mass""")
        return None
    else:
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


def find_tanimoto_sim(matches, documents_library, query_smiles):
    '''To each match in matches df, add the tanimoto similarity between query and match

    matches: pandas DataFrame, library matching result of 1 query on library
    documents_library: list of SpectrumDocuments, spectra in library
    df: pandas DataFrame, library matching result of 1 query on library with tanimoto similarities
    '''
    df = matches.copy()
    sims = []
    library_ids = df.index.values

    if not query_smiles or query_smiles == "None":  # check that query smiles exist
        df['similarity'] = [0] * len(
            library_ids)  # default to all 0 if it doesnt exist
        return df
    ms_q = Chem.MolFromSmiles(query_smiles)
    if not ms_q:  # in case something is wrong with smiles
        df['similarity'] = [0] * len(
            library_ids)  # default to all 0 if it doesnt exist
        return df

    fp_q = Chem.RDKFingerprint(ms_q)
    for lib_id in library_ids:
        smiles_lib = documents_library[lib_id]._obj.get("smiles")
        if smiles_lib and smiles_lib != "None":
            ms_lib = Chem.MolFromSmiles(smiles_lib)
            if ms_lib:
                fp_lib = Chem.RDKFingerprint(ms_lib)
                score = DataStructs.FingerprintSimilarity(fp_q, fp_lib)
            else:  # in case something is wrong with smiles
                score = 0
        else:  # in case it doesnt have smiles
            score = 0
        sims.append(score)
    df['similarity'] = sims
    return df


def transform_num_matches(input_df, exp=0.93):
    '''Transform the cosine_matches and mod_cosine_matches to between 0-1

    input_df: pandas DataFrame, spec2vec matches for one query
    exp: int, the base for the exponential, default: 0.93

    Both matches are transformed to between 0-1 by doing 1-0.93^num_matches
    '''
    df = input_df.copy()  # otherwise it edits the df outside the function
    df['cosine_matches'] = [(1 - 0.93 ** i) for i in df['cosine_matches']]
    df['mod_cosine_matches'] = [(1 - 0.93 ** i) for i in
                                df['mod_cosine_matches']]
    return df


def find_mass_similarity(matches, documents_library, query_mass, base_num=0.8):
    '''
    To each match in matches df, add a scaled value for how similar the parent_mass is to the query

    matches: pandas DataFrame, library matching result of 1 query on library
    documents_library: list of SpectrumDocuments, spectra in library
    query_mass: float, parent mass of query
    base_num: float, the base for the exponent
    df: pandas DataFrame, library matching result of 1 query on library with mass sims

    The similarity in dalton is calculated and transformed into a value 0 - 1 by doing
    1 - base_num^diff_in_dalton
    '''
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
