import pickle
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec

import pandas as pd
from typing import List
from matchms import Spectrum
from matplotlib import pyplot as plt
from spec2vec.vector_operations import cosine_similarity_matrix
from spec2vec.vector_operations import calc_vector
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model as load_ms2ds_model
from ms2query.app_helpers import load_pickled_file
from ms2query.spectrum_processing import create_spectrum_documents

def get_all_pos_of_match_for_s2v_preselection(query_spectra,
                                              s2v_model,
                                              s2v_embeddings,
                                              best_matching_spectra):
    all_positions_of_matches = []
    for spectrum in tqdm(query_spectra):
        sorted_scores = get_sorted_s2v_scores(spectrum,
                                              s2v_model,
                                              s2v_embeddings)
        best_matches = best_matching_spectra[spectrum.get("spectrumid")]
        position_of_matches = []
        for match in best_matches:
            position_of_matches.append(sorted_scores.index.get_loc(match))
        all_positions_of_matches.append(position_of_matches)
    return all_positions_of_matches

def get_all_pos_of_match_for_ms2ds_preselection(query_spectra,
                                ms2ds_model_file,
                                ms2ds_embeddings_file,
                                best_matching_spectra):
    ms2ds_model = load_ms2ds_model(ms2ds_model_file)
    ms2ds_embeddings = load_pickled_file(ms2ds_embeddings_file)
    all_positions_of_matches = []
    for spectrum in tqdm(query_spectra):
        sorted_scores = get_sorted_ms2ds_scores(spectrum,
                                                ms2ds_model,
                                                ms2ds_embeddings)
        best_matches = [best_matching_spectra[spectrum.get("spectrumid")]]
        position_of_matches = []
        for match in best_matches:
            position_of_matches.append(sorted_scores.index.get_loc(match))
        all_positions_of_matches.append(position_of_matches)
    return all_positions_of_matches

def bar_plot_result(positions_of_matches: List[List[int]]):
    positions_dict = {}
    bin_nr = 20
    for i in range(0, 120000, bin_nr):
        positions_dict[i] = 0
    for positions in positions_of_matches:
        best_position = min(positions)
        binned_pos = best_position//bin_nr*bin_nr
        positions_dict[binned_pos] += 1

    print("making_plot")
    plt.bar(list(positions_dict.keys()),
            list(positions_dict.values()),
            width=bin_nr)
    plt.xlabel("Rank in ordered ms2ds scores")
    plt.ylabel("Nr of best matches")
    plt.show()

def plot_graph(positions_of_matches: List[List[int]],
               only_best_position=True):
    positions_dict = {}
    # nr_of_query_spectra = len(positions_of_matches)
    nr_of_query_spectra = 0
    for i in range(0, 120000):
        positions_dict[i] = 0
    for positions in positions_of_matches:
        if not only_best_position:
            for position in positions:
                positions_dict[position] += 1
                nr_of_query_spectra += 1
        else:
            best_position = min(positions)
            positions_dict[best_position] += 1
            nr_of_query_spectra += 1

    percentage_of_best_matches = 0
    last_position = 0
    for pos in positions_dict:
        percentage_of_best_matches += positions_dict[pos]/nr_of_query_spectra*100
        if positions_dict[pos] != 0:
            last_position = pos
        positions_dict[pos] = percentage_of_best_matches
    print("making_plot")
    plt.plot(list(positions_dict.keys())[:last_position],
             list(positions_dict.values())[:last_position])
    plt.xlabel("Nr of top ms2ds scores selected")
    plt.ylabel("Accumulative percentage of best matches in selection (%)")
    plt.show()

def plot_parent_mass(library_spectra,
                     query_spectra,
                     best_matches):
    perc_found_list = []
    average_matches_list = []
    mass_tolerances = []
    for i in tqdm(range(20)):
        mass_tolerance = i/5

        percentage_found, average_nr_of_matches = efficient_select_similar_parent_mass(
            library_spectra,
            query_spectra,
            best_matches,
            mass_tolerance)
        perc_found_list.append(percentage_found * 100)
        average_matches_list.append(average_nr_of_matches)
        mass_tolerances.append(mass_tolerance)
    print(mass_tolerances)
    print(perc_found_list)
    print(average_matches_list)
    fig, axs = plt.subplots(2)
    axs[0].plot(mass_tolerances, perc_found_list)
    axs[1].plot(mass_tolerances, average_matches_list)
    axs[0].set_ylim([0, 100])
    axs[0].set_ylabel("Found matches (%)")
    axs[1].set_ylabel("Avg nr of found spectra")
    axs[1].set_xlabel("Mass tolerance")
    plt.show()

def select_on_parent_mass_and_ms2ds(library_spectra,
                                    query_spectra,
                                    mass_tolerance,
                                    ms2ds_embeddings,
                                    ms2ds_model,
                                    best_matches):
    parent_mass_dict = {}
    for library_spectrum in library_spectra:
        parent_mass_dict[library_spectrum.get("spectrumid")] = \
            library_spectrum.get("parent_mass")

    parent_mass_df = pd.DataFrame(list(parent_mass_dict.items()),
                                  columns=["spectrum_id", "parent_mass"])
    parent_mass_df.sort_values("parent_mass", inplace=True, ignore_index=True)
    all_found_matches = {}
    for query_spectrum in query_spectra:
        query_parent_mass = query_spectrum.get("parent_mass")
        query_spectrum_id = query_spectrum.get("spectrumid")
        lower_bound = parent_mass_df["parent_mass"].searchsorted(
            query_parent_mass - mass_tolerance)
        upper_bound = parent_mass_df["parent_mass"].searchsorted(
            query_parent_mass + mass_tolerance)
        spectrum_ids = list(
            parent_mass_df["spectrum_id"].iloc[lower_bound:upper_bound])
        all_found_matches[query_spectrum_id] = spectrum_ids

    ms2ds_cut_off_values = {}
    for i in range(100):
        ms2ds_cut_off_values[i/100] = [0, 0]

    for query_spectrum in tqdm(query_spectra):
        query_spectrum_id = query_spectrum.get("spectrumid")
        library_spectrum_ids = all_found_matches[query_spectrum_id]

        query_ms2ds_embeddings = MS2DeepScore(
            ms2ds_model,
            progress_bar=False).calculate_vectors([query_spectrum])
        preselected_ms2ds_embeddings = \
            ms2ds_embeddings.loc[library_spectrum_ids].to_numpy()
        ms2ds_scores = cosine_similarity_matrix(query_ms2ds_embeddings,
                                                preselected_ms2ds_embeddings
                                                )[0]

        for ms2ds_cutoff in ms2ds_cut_off_values:
            selected_spectra = []
            for i, ms2ds_score in enumerate(ms2ds_scores):
                library_spectrum_id = library_spectrum_ids[i]
                if ms2ds_score > ms2ds_cutoff:
                    selected_spectra.append(library_spectrum_id)
            ms2ds_cut_off_values[ms2ds_cutoff][1] += len(selected_spectra)
            actual_match = best_matches[query_spectrum_id]
            if actual_match in selected_spectra:
                ms2ds_cut_off_values[ms2ds_cutoff][0] += 1

    print(ms2ds_cut_off_values)



def efficient_select_similar_parent_mass(library_spectra,
                                         query_spectra,
                                         best_matches,
                                         mass_tolerance
                                         ):
    parent_mass_dict = {}
    for library_spectrum in library_spectra:
        parent_mass_dict[library_spectrum.get("spectrumid")] = \
            library_spectrum.get("parent_mass")

    parent_mass_df = pd.DataFrame(list(parent_mass_dict.items()),
                                  columns=["spectrum_id", "parent_mass"])
    # parent_mass_df.set_index("parent_mass", inplace=True)
    parent_mass_df.sort_values("parent_mass", inplace=True, ignore_index=True)
    all_found_matches = {}
    total_nr_of_found_matches = 0
    for query_spectrum in query_spectra:
        query_parent_mass = query_spectrum.get("parent_mass")
        query_spectrum_id = query_spectrum.get("spectrumid")
        lower_bound = parent_mass_df["parent_mass"].searchsorted(
            query_parent_mass - mass_tolerance - 0.000000000001)
        upper_bound = parent_mass_df["parent_mass"].searchsorted(
            query_parent_mass + mass_tolerance + 0.000000000001)
        spectrum_ids = list(
            parent_mass_df["spectrum_id"].iloc[lower_bound:upper_bound])
        all_found_matches[query_spectrum_id] = spectrum_ids
        total_nr_of_found_matches += len(spectrum_ids)

    nr_of_matches_found = 0
    for query_spectrum_id in best_matches:
        actual_match = best_matches[query_spectrum_id]
        found_matches = all_found_matches[query_spectrum_id]
        if actual_match in found_matches:
            nr_of_matches_found += 1
    percentage_found = nr_of_matches_found/len(best_matches)
    average_nr_of_matches = total_nr_of_found_matches/len(best_matches)
    return percentage_found, average_nr_of_matches


def selected_similar_parent_mass(library_spectra,
                                 query_spectra,
                                 best_matches,
                                 mass_tolerance):
    all_found_matches = {}
    total_nr_of_found_matches = 0
    for query_spectrum in tqdm(query_spectra):
        query_parent_mass = query_spectrum.get("parent_mass")
        query_spectrum_id = query_spectrum.get("spectrumid")
        all_found_matches[query_spectrum_id] = []
        for library_spectrum in library_spectra:
            library_parent_mass = library_spectrum.get("parent_mass")
            if query_parent_mass - mass_tolerance <= library_parent_mass <= \
                    query_parent_mass + mass_tolerance:

                all_found_matches[query_spectrum_id].append(
                    library_spectrum.get("spectrumid"))
                total_nr_of_found_matches += 1

    nr_of_matches_found = 0
    for query_spectrum_id in best_matches:
        actual_match = best_matches[query_spectrum_id]
        found_matches = all_found_matches[query_spectrum_id]
        if actual_match in found_matches:
            nr_of_matches_found += 1
    percentage_found = nr_of_matches_found/len(best_matches)
    average_nr_of_matches = total_nr_of_found_matches/len(best_matches)
    return percentage_found, average_nr_of_matches

def get_sorted_s2v_scores(query_spectrum,
                          s2v_model,
                          s2v_embeddings):

    # Convert list of Spectrum objects to list of SpectrumDocuments
    query_spectrum_document = create_spectrum_documents([query_spectrum])[0]

    query_embeddings = np.array([calc_vector(s2v_model,
                                             query_spectrum_document,
                                             allowed_missing_percentage=100)])

    library_embeddings_np = s2v_embeddings.to_numpy()
    s2v_scores = cosine_similarity_matrix(library_embeddings_np,
                                            query_embeddings)

    similarity_matrix_dataframe = pd.DataFrame(
        s2v_scores,
        index=s2v_embeddings.index,
        columns=["s2v_score"])
    similarity_matrix_dataframe.sort_values("s2v_score", inplace=True,
                                            ascending=False)
    return similarity_matrix_dataframe


def get_sorted_ms2ds_scores(query_spectrum: Spectrum,
                          ms2ds_model,
                          ms2ds_embeddings) -> pd.DataFrame:
    """Returns a dataframe with the ms2deepscore similarity scores

    query_spectrum
        Spectrum for which similarity scores should be calculated for all
        spectra in the ms2ds embeddings file.
    """
    ms2ds = MS2DeepScore(ms2ds_model, progress_bar=False)
    query_embedding = ms2ds.calculate_vectors([query_spectrum])
    library_ms2ds_embeddings_numpy = ms2ds_embeddings.to_numpy()

    ms2ds_scores = cosine_similarity_matrix(library_ms2ds_embeddings_numpy,
                                            query_embedding)
    similarity_matrix_dataframe = pd.DataFrame(
        ms2ds_scores,
        index=ms2ds_embeddings.index,
        columns=["ms2ds_score"])
    similarity_matrix_dataframe.sort_values("ms2ds_score", inplace=True,
                                            ascending=False)
    return similarity_matrix_dataframe

def get_best_matching_spectrum_ids_for_spectra(large_spectra_set,
                                               test_spectra_set,
                                               tanimoto_scores_file):
    all_inchikeys = get_all_inchikeys(large_spectra_set)
    inchikeys_with_spectra_id = sort_spectra_by_inchikey(large_spectra_set,
                                                         all_inchikeys)
    best_matching_inchikeys = find_all_best_matching_inchikeys(
        tanimoto_scores_file, test_spectra_set, all_inchikeys)

    best_matching_spectra = {}
    for query_spectrum_id in best_matching_inchikeys:
        best_matching_inchikey = best_matching_inchikeys[query_spectrum_id]
        best_matching_spectra[query_spectrum_id] = \
            inchikeys_with_spectra_id[best_matching_inchikey]
    return best_matching_spectra


def sort_spectra_by_inchikey(spectra, allowed_inchikeys):
    """Returns all spectra ids corresponding to an inchikey"""
    inchikey_dict = {}
    for inchikey in allowed_inchikeys:
        inchikey_dict[inchikey] = []
    for spectrum in tqdm(spectra):
        inchikey = spectrum.get("inchikey")[:14]
        spectrum_id = spectrum.get("spectrumid")
        if len(inchikey) == 14:
            inchikey_dict[inchikey].append(spectrum_id)
    return inchikey_dict


def get_all_inchikeys(spectra_list):
    """Returns all inchikeys in spectra_list"""
    inchikeys = {}
    for spectrum in spectra_list:
        inchikey = spectrum.get("inchikey")[:14]
        inchikeys[inchikey] = True
    inchikeys = list(inchikeys.keys())
    return inchikeys


def find_all_best_matching_inchikeys(tanimoto_scores_file,
                                     query_spectra,
                                     available_inchikeys):
    highest_inchikeys = {}
    for spectrum in tqdm(query_spectra,
                         desc="Finding inchikey with highest tanimoto"):
        found_inchikey = find_inchi_with_highest_tanimoto_score(
            tanimoto_scores_file,
            spectrum,
            available_inchikeys)
        highest_inchikeys[spectrum.get("spectrumid")] = found_inchikey
    return highest_inchikeys


def find_inchi_with_highest_tanimoto_score(tanimoto_scores_file: str,
                                           query_spectrum: Spectrum,
                                           available_inchikeys: List[str]
                                           ) -> str:
    """Returns the inchikey with the highest tanimoto score"""
    inchikey14 = query_spectrum.get("inchikey")[:14]
    assert len(inchikey14) == 14, "Expected an inchikey of length 14"

    all_tanimoto_scores = load_pickled_file(tanimoto_scores_file)
    tanimoto_scores = all_tanimoto_scores[inchikey14].loc[available_inchikeys]
    inchikey_with_highest_tanimoto = tanimoto_scores.idxmax()
    return inchikey_with_highest_tanimoto


def make_new_dataset(validation_spectra):
    """For every inchikey that has multiple spectra one spectrum is selected
    and returned as new_training_spectra. The other spectra of this inchikey
    are added together in kept_validation_spectra"""
    new_training_spectra = []
    kept_validation_spectra = []

    # Create list of all inchikeys
    inchikey_list = []
    for spectrum in validation_spectra:
        inchikey = spectrum.get("inchikey")[:14]
        inchikey_list.append(inchikey)
    inchikey_set = set(inchikey_list)
    # Initialize inchikey dict
    inchikey_dict = {}
    for inchikey in inchikey_set:
        inchikey_dict[inchikey] = []
    # Fill dictionary and split set into two
    for spectrum in validation_spectra:
        inchikey = spectrum.get("inchikey")[:14]
        inchikey_dict[inchikey].append(spectrum)
    best_spectrum_matches = {}
    for inchikey in inchikey_dict:
        spectra = inchikey_dict[inchikey]
        if len(spectra) > 1:
            new_training_spectra.append(spectra[0])
            for spectrum in spectra[1:]:
                kept_validation_spectra.append(spectrum)
                best_spectrum_matches[spectrum.get("spectrumid")] = spectra[0].get("spectrumid")
    return new_training_spectra, kept_validation_spectra, best_spectrum_matches

def plot_preselection_on_parent_mass_and_ms2ds(nr_of_query_spectra = 3094):
    """Plots the result when preselection is done for parent mass tolerance = 2
    """
    result = {0.0: [2886, 1863738], 0.01: [2886, 1863738],
              0.02: [2886, 1863733],
              0.03: [2886, 1863717], 0.04: [2886, 1863623],
              0.05: [2886, 1863343],
              0.06: [2886, 1862460], 0.07: [2886, 1860752],
              0.08: [2886, 1857999],
              0.09: [2886, 1853608], 0.1: [2886, 1845699],
              0.11: [2886, 1833967],
              0.12: [2886, 1820758], 0.13: [2886, 1805280],
              0.14: [2885, 1786024],
              0.15: [2885, 1762957], 0.16: [2885, 1736203],
              0.17: [2884, 1706812],
              0.18: [2884, 1676722], 0.19: [2879, 1645689],
              0.2: [2876, 1613215],
              0.21: [2873, 1578764], 0.22: [2871, 1542966],
              0.23: [2865, 1505873],
              0.24: [2861, 1467682], 0.25: [2860, 1428168],
              0.26: [2855, 1387847],
              0.27: [2850, 1347521], 0.28: [2843, 1306577],
              0.29: [2840, 1265461],
              0.3: [2837, 1224504], 0.31: [2834, 1183520],
              0.32: [2830, 1143384],
              0.33: [2812, 1102860], 0.34: [2795, 1062585],
              0.35: [2787, 1022937],
              0.36: [2780, 983488], 0.37: [2765, 944736], 0.38: [2749, 906338],
              0.39: [2745, 868404], 0.4: [2739, 830934], 0.41: [2725, 794114],
              0.42: [2713, 758348], 0.43: [2702, 724190], 0.44: [2692, 690876],
              0.45: [2683, 658614], 0.46: [2675, 627570], 0.47: [2661, 597173],
              0.48: [2642, 567965], 0.49: [2627, 539537], 0.5: [2607, 512662],
              0.51: [2582, 487043], 0.52: [2563, 462356], 0.53: [2545, 438379],
              0.54: [2518, 415659], 0.55: [2497, 393952], 0.56: [2469, 373265],
              0.57: [2451, 352990], 0.58: [2434, 333812], 0.59: [2400, 315487],
              0.6: [2375, 298279], 0.61: [2341, 281559], 0.62: [2317, 265462],
              0.63: [2294, 250452], 0.64: [2261, 236017], 0.65: [2231, 222079],
              0.66: [2204, 208897], 0.67: [2168, 196644], 0.68: [2135, 184836],
              0.69: [2101, 173840], 0.7: [2072, 163357], 0.71: [2035, 153407],
              0.72: [2000, 144236], 0.73: [1959, 135822], 0.74: [1915, 127779],
              0.75: [1877, 120496], 0.76: [1837, 113429], 0.77: [1791, 106835],
              0.78: [1737, 100396], 0.79: [1675, 94307], 0.8: [1612, 88516],
              0.81: [1553, 82940], 0.82: [1505, 77399], 0.83: [1458, 71640],
              0.84: [1400, 64502], 0.85: [1333, 55631], 0.86: [1256, 47565],
              0.87: [1172, 41942], 0.88: [1126, 38104], 0.89: [1069, 34782],
              0.9: [992, 31628], 0.91: [900, 28192], 0.92: [844, 25340],
              0.93: [765, 22631], 0.94: [689, 19357], 0.95: [607, 15944],
              0.96: [505, 12790], 0.97: [408, 9942], 0.98: [317, 7203],
              0.99: [222, 3471]}
    ms2ds_tolerances = []
    perc_found_list = []
    average_matches_list = []
    for key in result:
        ms2ds_tolerances.append(key)
        perc_found_list.append(result[key][0] / nr_of_query_spectra * 100)
        average_matches_list.append(result[key][1] / nr_of_query_spectra)

    fig, axs = plt.subplots(2)
    axs[0].plot(ms2ds_tolerances, perc_found_list)
    axs[1].plot(ms2ds_tolerances, average_matches_list)
    axs[0].set_ylim([0, 100])
    axs[0].set_ylabel("Found matches (%)")
    axs[1].set_ylabel("Avg nr of found spectra")
    axs[1].set_xlabel("Minimal ms2ds score")
    plt.show()

def plot_parent_mass_occurances(spectra):
    bins = {}
    step_size = 0.1
    for i in range(40000):
        i = i * step_size
        bins[round(i, 1)] = 0
    for spectrum in spectra:
        parent_mass = spectrum.get("parent_mass")
        parent_mass = int(parent_mass*10)/10
        bins[parent_mass] += 1

    plt.bar(list(bins.keys())[9000:10000],
            list(bins.values())[9000:10000],
            width=step_size)
    plt.xlabel("parent_mass")
    plt.ylabel("Nr of entries")
    plt.show()

if __name__ == "__main__":
    validation_spectra = load_pickled_file("../downloads/optimizing_preselection/library_with_matching_inchi/validation_spectra_with_inchikey_match.pickle")
    library_spectra = load_pickled_file("../downloads/optimizing_preselection/library_with_matching_inchi/library_spectra_with_matching_inchi.pickle")
    # ms2ds_model_file = "../downloads/train_ms2query_nn_data/ms2ds/ms2ds_siamese_210301_5000_500_400.hdf5"
    # ms2ds_model = load_ms2ds_model(ms2ds_model_file)
    # ms2ds_embeddings_file = "../downloads/optimizing_preselection/library_with_matching_inchi/ms2ds_embeddings_library_with_matching_inchi.pickle"
    # ms2ds_embeddings = load_pickled_file(ms2ds_embeddings_file)
    best_matching_spectra = load_pickled_file("../downloads/optimizing_preselection/library_with_matching_inchi/best_matching_spectra_with_matching_inchi.pickle")
    # s2v_model = Word2Vec.load("../downloads/train_ms2query_nn_data/spec2vec_model/ALL_GNPS_positive_210305_Spec2Vec_strict_filtering_iter_20.model")
    # s2v_embeddings = load_pickled_file("../downloads/train_ms2query_nn_data/spec2vec_model/ALL_GNPS_positive_train_split_210305_s2v_embeddings.pickle")
    # result = get_all_pos_of_match_for_s2v_preselection(validation_spectra,
    #                                                    s2v_model,
    #                                                    s2v_embeddings,
    #                                                    best_matching_spectra)
    s2v_match_pos = load_pickled_file("../downloads/optimizing_preselection/matches_position_for_s2v_preselection")
    ms2ds_match_pos = load_pickled_file("../downloads/optimizing_preselection/matches_position_for_ms2ds_preselection.pickle")
    # plot_graph(ms2ds_match_pos, only_best_position=False)

    # plot_parent_mass(library_spectra,
    #                  validation_spectra,
    #                  best_matching_spectra)

