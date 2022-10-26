import os
from typing import List, Tuple, Union
from ms2query.run_ms2query import run_complete_folder
from ms2query.ms2library import MS2Library
import sqlite3
import pandas as pd
from tqdm.notebook import tqdm
from ms2deepscore.models import load_model as load_ms2ds_model
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import SiameseModel
from spec2vec.vector_operations import calc_vector, cosine_similarity_matrix
from ms2query.ms2library import create_library_object_from_one_dir
from ms2query.utils import save_pickled_file
from ms2query.query_from_sqlite_database import get_metadata_from_sqlite
from ms2query.create_new_library.train_ms2query_model import calculate_tanimoto_scores


def generate_test_results_ms2query(ms2library: MS2Library,
                                   test_spectra,
                                   temporary_file_csv_results):
    assert not os.path.isfile(temporary_file_csv_results), "file already exists"
    ms2library.analog_search_store_in_csv(test_spectra,
                                          results_csv_file_location=temporary_file_csv_results)
    df_results_ms2query = pd.read_csv(temporary_file_csv_results)
    os.remove(temporary_file_csv_results)
    test_results_ms2query = []
    for spectrum_id, ms2query_model_prediction, query_spectrum_nr in df_results_ms2query[
        ["spectrum_ids", "ms2query_model_prediction", "query_spectrum_nr"]].to_numpy():
        test_spectrum = test_spectra[query_spectrum_nr - 1]
        tanimoto_score = calculate_tanimoto_scores(ms2library.sqlite_file_name,
                                                   test_spectrum,
                                                   [spectrum_id]).iloc[0, 0]
        test_results_ms2query.append([spectrum_id, ms2query_model_prediction, tanimoto_score, test_spectrum])
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
                                       allowed_mass_diff):
    highest_ms2_deepscore_in_mass_range = []
    for i, spectrum in tqdm(enumerate(test_spectra)):
        precursor_mz_query_spectrum = spectrum.get("precursor_mz")
        spectra_and_mass = get_precursor_mz_within_range(sqlite_file_location,
                                                         precursor_mz_query_spectrum-allowed_mass_diff,
                                                         precursor_mz_query_spectrum+allowed_mass_diff)
        spectra = [spectrum_and_mass[0] for spectrum_and_mass in spectra_and_mass]
        highest_ms2_deepscore_in_mass_range.append(ms2deepscores[i].loc[spectra].idxmax())
    results_ms2deepscore = [(highest_ms2_deepscore_in_mass_range[i][0],
                             highest_ms2_deepscore_in_mass_range[i][1], test_spectra[i]) for i in
                            range(len(test_spectra))]
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


def generate_test_results(folder_with_models,
                          training_spectra,
                          test_spectra,
                          output_folder):
    ms2library = create_library_object_from_one_dir(folder_with_models)

    # Output of all tools is in the format: [lib_spec_id_highest_score, predicted_score, test_spectrum]
    # Generate MS2Query results
    ms2query_test_results = generate_test_results_ms2query(ms2library,
                                                           test_spectra,
                                                           os.path.join(output_folder, "temporary_ms2query_results.csv"))
    save_pickled_file(ms2query_test_results,
                      os.path.join(output_folder, "ms2query_test_results.pickle"))
    # Generate MS2Deepscore results
    ms2ds_scores = get_all_ms2ds_scores(ms2library.ms2ds_model,
                                        ms2library.ms2ds_embeddings,
                                        test_spectra)

    sqlite_file_name = ms2library.sqlite_file_name
    ms2ds_test_results = select_highest_ms2ds_in_mass_range(ms2ds_scores,
                                                            test_spectra,
                                                            ms2library.sqlite_file_name,
                                                            allowed_mass_diff=100)
    save_pickled_file(ms2ds_test_results,
                      os.path.join(output_folder, "ms2deepscore_test_results.pickle"))

    # Generate Modified cosine results
