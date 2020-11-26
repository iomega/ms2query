from rdkit import Chem, DataStructs


def find_info_matches(matches, documents_library, documents_query,
                      add_label=True, add_tanimoto_sim=True, add_cols=False,
                      add_num_matches_transform=True, add_mass_transform=True,
                      max_parent_mass=False, add_mass_similarity=True):
    '''
    To each df in matches, add more info like tanimoto scores vs query etc.

    A matching inchikey gets label 1 and non-matching 0.

    matches: list of pandas DataFrames, library matching result of query on library
    documents_library: list of SpectrumDocuments, spectra in library
    documents_query: list of SpectrumDocuments, spectra in query set.
        Indices should correspond to indices of matches.
    bins: list of int/float, the cutoff for the mass change, will result in boolean columns,
        i.e. if a cutoff is 2 there will be a column with 1 (true) or 0 (false) for if the mass change
        is within 2
    calc_change: bool, instead of bins add a percentage change of the parent mass difference
    add_label: bool, add a label for inchikey match or not
    add_tanimoto_sim: bool, add tanimoto similarity or not
    add_cols: bool/list of str, add other info present in metadata such as parent_mass, adduct
        default: False
    add_num_matches_transform: bool, transform cosine and mod_cosine matches to a number 0-1.
        both matches are transformed to between 0-1 by doing 1-0.93^num_matches
    add_mass_transform: bool, add transform of the parent masses to a fraction of the maximal parent mass
    max_parent_mass: int/float, the maximum parent mass in the dataset, default = False
    add_mass_similarity: bool, add similarity of parent mass to the query as a scaled number from 0-1
        where The similarity in dalton is calculated and transformed into a value 0 - 1 by doing
        1 - base_num^diff_in_dalton

    Output
    matches_with_info: list of pandas DataFrames, library matching result of query
        on library with matching labels
    '''
    matches_with_info = []
    if add_mass_transform and not max_parent_mass:
        print(
            "If you want to transform the masses, please provide a max_parent_mass")
        return
    else:
        print('Max parent mass:', max_parent_mass)

    for query_id in range(len(documents_query)):
        match = matches[query_id].copy()
        if add_label:
            query_inchi = documents_query[query_id]._obj.get("inchikey")[:14]
            match = find_inchikey_match(match, documents_library, query_inchi)
        if add_cols:
            match = find_basic_info(match, documents_library, add_cols)
        if add_tanimoto_sim:
            query_smiles = documents_query[query_id]._obj.get("smiles")
            match = find_tanimoto_sim(match, documents_library, query_smiles)
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
            q_mass = documents_query[query_id]._obj.get("parent_mass")
            match = find_mass_similarity(match, documents_library, q_mass,
                                         base_num=0.8)

        matches_with_info.append(match)
    return matches_with_info


def find_inchikey_match(matches, documents_library, query_inchi):
    '''To each match in the matches df, add label for matching inchikey (1) or non-matching (0)

    matches: pandas DataFrame, library matching result of 1 query on library
    documents_library: list of SpectrumDocuments, spectra in library
    query_inchi: str, first 14 symbols of query inchikey
    df: pandas DataFrame, library matching result of query on library with matching labels
    '''
    matches_with_labels = []
    df = matches.copy()
    library_ids = df.index.values
    labels = []
    for lib_id in library_ids:
        lib_inchi = documents_library[lib_id]._obj.get("inchikey")[:14]
        lab = 0
        if query_inchi == lib_inchi:
            lab = 1
        labels.append(lab)
    df['label'] = labels
    return df


def find_mass_matches(matches, documents_library, query_mass, bins=[2, 5],
                      calc_change=False):
    '''
    To each match in matches df, add the mass change of query to match in bins or a percentage change

    matches: pandas DataFrame, library matching result of 1 query on library
    documents_library: list of SpectrumDocuments, spectra in library
    query_mass: float, parent mass of query
    bins: list of int/float, the cutoff for the mass change, will result in boolean columns,
        i.e. if a cutoff is 2 there will be a column with 1 (true) or 0 (false) for if the mass change
        is within 2
    calc_change: bool, instead of bins add a percentage change of the parent mass difference
    df: pandas DataFrame, library matching result of 1 query on library with mass matches
    '''
    range_bins = range(len(bins))  # calc once
    df = matches.copy()
    library_ids = df.index.values
    masses = [[] for _ in range_bins]  # initialise
    mass_changes = []
    for lib_id in library_ids:
        lib_mass = documents_library[lib_id]._obj.get("parent_mass")
        if calc_change:  # calculate a percentage change in parent mass instead of discrete bins
            perc_change = abs(lib_mass - query_mass) / query_mass * 100
            mass_changes.append(perc_change)
        else:
            for bin_i in range_bins:
                cutoff = bins[bin_i]
                lab = 0
                if abs(query_mass - lib_mass) < cutoff:
                    lab = 1
                masses[bin_i].append(lab)
    # add to df
    if calc_change:
        df['perc_mass_change'] = mass_changes
    else:
        for bin_i in range_bins:
            df['mass_match_' + str(bins[bin_i])] = masses[bin_i]
    return df


def find_basic_info(matches, documents_library, add_cols=['parent_mass']):
    '''
    To each match in matches df, add the info from add_cols entries

    matches: pandas DataFrame, library matching result of 1 query on library
    documents_library: list of SpectrumDocuments, spectra in library
    df: pandas DataFrame, library matching result of 1 query on library with added info
    '''
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
