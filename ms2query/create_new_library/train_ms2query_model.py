"""
This script is not needed for normally running MS2Query, it is only needed to to train
new models
"""

import os
from typing import List
import pandas as pd
from tqdm import tqdm
from matchms import Spectrum
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from ms2query import MS2Library
from ms2query.query_from_sqlite_database import get_metadata_from_sqlite
from ms2query.create_new_library.library_files_creator import LibraryFilesCreator
from ms2query.create_new_library.split_data_for_training import split_spectra_on_inchikeys, split_training_and_validation_spectra
from ms2query.create_new_library.calculate_tanimoto_scores import calculate_tanimoto_scores_from_smiles
from ms2query.utils import save_pickled_file


class DataCollectorForTraining():
    """Class to collect data needed to train a ms2query random forest"""
    def __init__(self,
                 ms2library: MS2Library,
                 preselection_cut_off: int = 100):
        """Parameters
        ----------
        ms2library:
            A ms2library object. These should not contain the query spectra.
        preselection_cut_off:
            The number of highest scoring matches of MS2Deepscore that are used. For these top library matches all
            scores are calculated
        """
        self.ms2library = ms2library
        self.preselection_cut_off = preselection_cut_off

    def get_matches_info_and_tanimoto(self,
                                      query_spectra: List[Spectrum]):
        """Returns tanimoto scores and info about matches of all query spectra

        A selection of matches is made for each query_spectrum. Based on the
        spectra multiple scores are calculated (info_of_matches_with_tanimoto)
        and the tanimoto scores based on the smiles is returned. All matches of
        all validation_query_spectra are added together and the order of the tanimoto
        scores corresponds to the order of the info, so they can be used for
        training.

        Args:
        ------
        validation_query_spectra:
            List of Spectrum objects
        """
        all_tanimoto_scores = pd.DataFrame()
        info_of_matches_with_tanimoto = pd.DataFrame()
        for query_spectrum in tqdm(query_spectra,
                                   desc="Get scores and tanimoto scores",
                                   disable=not self.ms2library.settings["progress_bars"]):
            results_table = self.ms2library.calculate_features_single_spectrum(query_spectrum,
                                                                               self.preselection_cut_off)

            # Get tanimoto scores
            library_spectrum_ids = list(results_table.data.index)
            tanimoto_scores = calculate_tanimoto_scores_with_library(self.ms2library.sqlite_file_name, query_spectrum,
                                                                     library_spectrum_ids)
            # Add tanimoto scores for training data
            all_tanimoto_scores = \
                all_tanimoto_scores.append(tanimoto_scores,
                                           ignore_index=True)
            # Select the features (remove inchikey, since this should not be
            # used for training
            features_dataframe = results_table.get_training_data()
            # Add matches for which a tanimoto score could be calculated
            matches_with_tanimoto = features_dataframe.loc[
                tanimoto_scores.index]
            info_of_matches_with_tanimoto = \
                info_of_matches_with_tanimoto.append(matches_with_tanimoto,
                                                     ignore_index=True)
        return info_of_matches_with_tanimoto, all_tanimoto_scores


def calculate_tanimoto_scores_with_library(sqlite_file_name,
                                           query_spectrum: Spectrum,
                                           spectra_ids_list: List[str]):
    # Get inchikeys belonging to spectra ids
    metadata_dict = get_metadata_from_sqlite(
        sqlite_file_name,
        spectra_ids_list)
    library_smiles_list = [metadata_dict[spectrum_id]["smiles"] for spectrum_id in spectra_ids_list]
    tanimoto_scores = calculate_tanimoto_scores_from_smiles(library_smiles_list, [query_spectrum.get("smiles")])
    tanimoto_df = pd.DataFrame(tanimoto_scores, index=spectra_ids_list, columns=["Tanimoto_score"])
    return tanimoto_df


def train_random_forest(selection_of_training_scores,
                        training_labels):
    # train rf using optimised parameters from below
    rf = RandomForestRegressor(n_estimators=250,
                               random_state=42,
                               max_depth=5,
                               min_samples_leaf=50,
                               n_jobs=7)
    rf.fit(selection_of_training_scores, training_labels)

    # predict on train
    rf_train_predictions = rf.predict(selection_of_training_scores)
    mse_train_rf = mean_squared_error(training_labels, rf_train_predictions)
    print('Training MSE', mse_train_rf)
    return rf


def train_ms2query_model(training_spectra,
                         library_files_folder,
                         ms2ds_model_file_name,
                         s2v_model_file_name,
                         fraction_for_training):
    # Select spectra belonging to a single InChIKey
    library_spectra, unique_inchikey_query_spectra = split_spectra_on_inchikeys(training_spectra,
                                                                                fraction_for_training)
    # Select random spectra from the library
    library_spectra, single_spectra_query_spectra = split_training_and_validation_spectra(library_spectra,
                                                                                          fraction_for_training)
    query_spectra_for_training = unique_inchikey_query_spectra + single_spectra_query_spectra

    # Create library files for training ms2query
    library_creator_for_training = LibraryFilesCreator(library_spectra, output_directory=library_files_folder,
                                                       s2v_model_file_name=s2v_model_file_name,
                                                       ms2ds_model_file_name=ms2ds_model_file_name)
    library_creator_for_training.create_all_library_files()

    ms2library_for_training = MS2Library(
        sqlite_file_name=library_creator_for_training.sqlite_file_name,
        s2v_model_file_name=s2v_model_file_name,
        ms2ds_model_file_name=ms2ds_model_file_name,
        pickled_s2v_embeddings_file_name=library_creator_for_training.s2v_embeddings_file_name,
        pickled_ms2ds_embeddings_file_name=library_creator_for_training.ms2ds_embeddings_file_name,
        ms2query_model_file_name=None,
        classifier_csv_file_name=None)
    # Create training data MS2Query model
    collector = DataCollectorForTraining(ms2library_for_training)
    training_scores, training_labels = collector.get_matches_info_and_tanimoto(query_spectra_for_training)

    save_pickled_file(training_scores, os.path.join(library_files_folder, "training_scores_ms2query"))
    save_pickled_file(training_labels, os.path.join(library_files_folder, "training_labels_ms2query"))

    # Train MS2Query model
    ms2query_model = train_random_forest(training_scores, training_labels)
    return ms2query_model
