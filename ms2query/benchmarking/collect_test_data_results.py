import os
from typing import List, Tuple, Union
from tqdm import tqdm
import random
from matchms import Spectrum
from ms2query.create_new_library.calculate_tanimoto_scores import calculate_single_tanimoto_score, calculate_tanimoto_scores_unique_inchikey
from ms2query.ms2library import MS2Library
import sqlite3
import pandas as pd
from tqdm.notebook import tqdm
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import SiameseModel
from spec2vec.vector_operations import cosine_similarity_matrix
from matchms.calculate_scores import calculate_scores
from matchms.similarity.ModifiedCosine import ModifiedCosine
from matchms.similarity.CosineGreedy import CosineGreedy

from ms2query.ms2library import create_library_object_from_one_dir
from ms2query.utils import save_json_file
from ms2query.query_from_sqlite_database import get_metadata_from_sqlite


def generate_test_results_ms2query(ms2library: MS2Library,
                                   test_spectra,
                                   temporary_file_csv_results) -> List[Tuple[float, float, bool]]:
    assert not os.path.isfile(temporary_file_csv_results), "file already exists"
    ms2library.analog_search_store_in_csv(test_spectra,
                                          results_csv_file_location=temporary_file_csv_results)
    df_results_ms2query = pd.read_csv(temporary_file_csv_results)
    os.remove(temporary_file_csv_results)
    test_results_ms2query = []
    for spectrum_id, ms2query_model_prediction, query_spectrum_nr in df_results_ms2query[
        ["spectrum_ids", "ms2query_model_prediction", "query_spectrum_nr"]].to_numpy():
        test_spectrum = test_spectra[int(query_spectrum_nr) - 1]
        # Get metadata belonging to spectra ids
        lib_metadata = get_metadata_from_sqlite(
            ms2library.sqlite_file_name,
            [spectrum_id])[spectrum_id]
        tanimoto_score = calculate_single_tanimoto_score(test_spectrum.get("smiles"), lib_metadata["smiles"])
        exact_match = lib_metadata["inchikey"][:14] == test_spectrum.get("inchikey")[:14]
        test_results_ms2query.append((ms2query_model_prediction, tanimoto_score, exact_match))
    return test_results_ms2query


def get_all_ms2ds_scores(ms2ds_model: SiameseModel,
                         ms2ds_embeddings,
                         test_spectra
                         ) -> pd.DataFrame:
    """Returns a dataframe with the ms2deepscore similarity scores

    The similarity scores are calculated between the query_spectra and all
    library spectra.

    query_spectra
        Spectra for which similarity scores should be calculated for all
        spectra in the ms2ds embeddings file.
    """
    # ms2ds_model = load_ms2ds_model(ms2ds_model_file_name)
    ms2ds = MS2DeepScore(ms2ds_model, progress_bar=False)
    query_embeddings = ms2ds.calculate_vectors(test_spectra)
    library_ms2ds_embeddings_numpy = ms2ds_embeddings.to_numpy()

    ms2ds_scores = cosine_similarity_matrix(library_ms2ds_embeddings_numpy,
                                            query_embeddings)
    similarity_matrix_dataframe = pd.DataFrame(
        ms2ds_scores,
        index=ms2ds_embeddings.index)
    return similarity_matrix_dataframe


def select_highest_ms2ds_in_mass_range(ms2deepscores,
                                       test_spectra,
                                       sqlite_file_location,
                                       allowed_mass_diff: Union[None, float]) -> List[Tuple[float, float, bool]]:
    results_ms2deepscore = []
    for i, test_spectrum in tqdm(enumerate(test_spectra)):
        precursor_mz_query_spectrum = test_spectrum.get("precursor_mz")
        if allowed_mass_diff is not None:
            spectrum_ids_and_mass = get_precursor_mz_within_range(sqlite_file_location,
                                                                  precursor_mz_query_spectrum - allowed_mass_diff,
                                                                  precursor_mz_query_spectrum + allowed_mass_diff)
            spectrum_ids = [spectrum_and_mass[0] for spectrum_and_mass in spectrum_ids_and_mass]
            spectrum_id_highest_ms2_deepscore_in_mass_range = ms2deepscores[i].loc[spectrum_ids].idxmax()
        else:
            spectrum_id_highest_ms2_deepscore_in_mass_range = ms2deepscores[i].idxmax()
        # Get metadata belonging to spectra ids
        lib_metadata = get_metadata_from_sqlite(
            sqlite_file_location,
            [spectrum_id_highest_ms2_deepscore_in_mass_range])[spectrum_id_highest_ms2_deepscore_in_mass_range]
        tanimoto_score = calculate_single_tanimoto_score(test_spectrum.get("smiles"), lib_metadata["smiles"])
        exact_match = lib_metadata["inchikey"][:14] == test_spectrum.get("inchikey")[:14]
        results_ms2deepscore.append((float(ms2deepscores[i][spectrum_id_highest_ms2_deepscore_in_mass_range]),
                                     tanimoto_score,
                                     exact_match))
    return results_ms2deepscore


def get_precursor_mz_within_range(sqlite_file_name: str,
                                  lower_bound: Union[float, int],
                                  upper_bound: Union[float, int],
                                  ) -> List[Tuple[str, float]]:
    """Returns spectrum_ids with precursor m/z between lower and upper bound

    Args:
    -----
    sqlite_file_name:
        The sqlite file in which the spectra data is stored.
    lower_bound:
        The lower bound of the allowed precursor m/z
    upper_bound:
        The upper bound of the allowed precursor m/z
    """
    conn = sqlite3.connect(sqlite_file_name)
    sqlite_command = \
        f"""SELECT spectrumid, precursor_mz FROM spectrum_data 
        WHERE precursor_mz BETWEEN {lower_bound} and {upper_bound}"""
    cur = conn.cursor()
    cur.execute(sqlite_command)
    spectrum_ids_within_range = cur.fetchall()
    return spectrum_ids_within_range


def select_spectra_within_mass_range(spectra, lower_bound, upper_bound):
    selected_spectra = []
    for spectrum in spectra:
        precursor_mz = spectrum.get('precursor_mz')
        if precursor_mz <= upper_bound and precursor_mz >= lower_bound:
            selected_spectra.append(spectrum)
    return selected_spectra


def get_modified_cosine_score_results(lib_spectra,
                                      test_spectra,
                                      mass_tolerance=100) -> List[Tuple[float, float, bool]]:
    best_matches_for_test_spectra = []
    for test_spectrum in tqdm(test_spectra):
        precursor_mz = test_spectrum.get("precursor_mz")
        selected_lib_spectra = select_spectra_within_mass_range(lib_spectra,
                                                                precursor_mz-mass_tolerance,
                                                                precursor_mz+mass_tolerance)
        if len(selected_lib_spectra) != 0:
            scores_list = calculate_scores(selected_lib_spectra,
                                           [test_spectrum], ModifiedCosine()).scores_by_query(test_spectrum)
            # Scores list is a List[spectrum, (mod_cos, matching_peaks)
            cosine_scores = [scores_tuple[1][0] for scores_tuple in scores_list]
            highest_cosine_score = max(cosine_scores)
            highest_scoring_spectrum = scores_list[cosine_scores.index(highest_cosine_score)][0]

            tanimoto_score = calculate_single_tanimoto_score(test_spectrum.get("smiles"),
                                                             highest_scoring_spectrum.get("smiles"))
            exact_match = highest_scoring_spectrum.get("inchikey")[:14] == test_spectrum.get("inchikey")[:14]
            best_matches_for_test_spectra.append((highest_cosine_score, tanimoto_score, exact_match))
        else:
            best_matches_for_test_spectra.append(None)
    return best_matches_for_test_spectra


def get_cosines_score_results(lib_spectra,
                              test_spectra,
                              mass_tolerance,
                              fragment_mass_tolerance,
                              minimum_matched_peaks):
    best_matches_for_test_spectra = []
    for test_spectrum in tqdm(test_spectra):
        precursor_mz = test_spectrum.get("precursor_mz")
        selected_lib_spectra = select_spectra_within_mass_range(lib_spectra, precursor_mz-mass_tolerance, precursor_mz+mass_tolerance)
        if len(selected_lib_spectra) != 0:
            scores_list = calculate_scores(selected_lib_spectra, [test_spectrum], CosineGreedy(tolerance=fragment_mass_tolerance)).scores_by_query(test_spectrum)
            cosine_scores = [scores_tuple[1].item()[0] for scores_tuple in scores_list if scores_tuple[1].item()[1] >= minimum_matched_peaks]
            if len(cosine_scores) != 0:
                highest_cosine_score = max(cosine_scores)
                highest_scoring_spectrum = scores_list[cosine_scores.index(highest_cosine_score)][0]

                tanimoto_score = calculate_single_tanimoto_score(test_spectrum.get("smiles"),
                                                                 highest_scoring_spectrum.get("smiles"))
                exact_match = highest_scoring_spectrum.get("inchikey")[:14] == test_spectrum.get("inchikey")[:14]
                best_matches_for_test_spectra.append((highest_cosine_score, tanimoto_score, exact_match))
            else:
                best_matches_for_test_spectra.append(None)
        else:
            best_matches_for_test_spectra.append(None)
    return best_matches_for_test_spectra


def create_optimal_results(test_spectra, training_spectra):
    smiles_test_spectra = [spectrum.get("smiles") for spectrum in test_spectra]
    smiles_training_spectra = [spectrum.get("smiles") for spectrum in training_spectra]
    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(smiles_test_spectra, smiles_training_spectra)


    inchikeys = [spectrum.get("inchikey")[:14] for spectrum in test_spectra]
    unique_inchikeys = list(set(inchikeys))


    # remove inchikeys in test spectra from dataframe
    tanimoto_scores_removed = tanimoto_scores[~tanimoto_scores.index.isin(unique_inchikeys)]
    highest_tanimoto_list = []
    for inchikey in inchikeys:
        highest_tanimoto_score = tanimoto_scores_removed[inchikey].max()
        highest_tanimoto_list.append((inchikey, highest_tanimoto_score, highest_tanimoto_score, inchikey))

    return highest_tanimoto_list


def create_random_results(test_spectra: List[Spectrum],
                          training_spectra: List[Spectrum]) -> List[Tuple[float, float, bool]]:
    random_predictions = []
    for test_spectrum in test_spectra:
        random_lib_spectrum = random.choice(training_spectra)
        tanimoto_score = calculate_single_tanimoto_score(test_spectrum.get("smiles"), random_lib_spectrum.get("smiles"))
        exact_match = random_lib_spectrum.get("inchikey")[:14] == test_spectrum.get("inchikey")[:14]
        random_predictions.append((random.random(), tanimoto_score, exact_match))
    return random_predictions


def generate_test_results(folder_with_models,
                          training_spectra,
                          test_spectra,
                          output_folder):
    assert os.path.isdir(output_folder)
    ms2library = create_library_object_from_one_dir(folder_with_models)
    # todo add assert statements to test that training spectra is matching the sqlite file

    # Output of all tools is in the format: [lib_spec_id_highest_score, predicted_score, test_spectrum]
    # Generate MS2Query results
    # todo replace temporary file name with a temp dir (no risk of accidentally storing)
    ms2query_test_results = generate_test_results_ms2query(
        ms2library,
        test_spectra,
        os.path.join(output_folder, "temporary_ms2query_results.csv"))

    # Generate MS2Deepscore results
    ms2ds_scores = get_all_ms2ds_scores(ms2library.ms2ds_model,
                                        ms2library.ms2ds_embeddings,
                                        test_spectra)

    ms2ds_test_results = select_highest_ms2ds_in_mass_range(ms2ds_scores,
                                                            test_spectra,
                                                            ms2library.sqlite_file_name,
                                                            allowed_mass_diff=100)

    # Generate Modified cosine results
    modified_cosine_results = get_modified_cosine_score_results(training_spectra, test_spectra, mass_tolerance=100)

    cosine_results = get_cosines_score_results(training_spectra,
                                               test_spectra,
                                               mass_tolerance=100,
                                               fragment_mass_tolerance=0.25,
                                               minimum_matched_peaks=0)


    # store as json file
    save_json_file({"ms2deepscore": ms2ds_test_results,
                    "modified cosine score": modified_cosine_results,
                    "cosine_score": cosine_results,
                    "ms2query": ms2query_test_results},
                   os.path.join(output_folder, "test_results.json"))


if __name__ == "__main__":
    from ms2query.utils import load_matchms_spectrum_objects_from_file

    training_spectra = load_matchms_spectrum_objects_from_file("../../data/test_dir/test_train_all_models/training_spectra_used/cleaned_training_spectra.pickle")
    test_spectra = load_matchms_spectrum_objects_from_file("../../data/test_dir/test_train_all_models/training_spectra_used/cleaned_test_spectra.pickle")

    generate_test_results("../../data/test_dir/test_train_all_models",
                          training_spectra,
                          test_spectra,
                          "../../data/test_dir/test_generate_test_results")
