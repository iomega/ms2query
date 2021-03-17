from typing import List, Dict, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model as load_nn_model
from gensim.models import Word2Vec
from matchms.similarity import CosineGreedy, ModifiedCosine
from matchms.Spectrum import Spectrum
from ms2deepscore.models import load_model as load_ms2ds_model
from ms2deepscore import MS2DeepScore
from spec2vec import Spec2Vec
from spec2vec.vector_operations import cosine_similarity_matrix
from ms2query.query_from_sqlite_database import get_spectra_from_sqlite
from ms2query.app_helpers import load_pickled_file
from ms2query.spectrum_processing import create_spectrum_documents


class Ms2Library:
    """Calculates scores of spectra in library and selects best matches"""
    def __init__(self,
                 sqlite_file_location: str,
                 s2v_model_file_name: str,
                 ms2ds_model_file_name: str,
                 pickled_s2v_embeddings_file_name: str,
                 pickled_ms2ds_embeddings_file_name: str,
                 **settings):
        """
        Parameters
        ----------
        sqlite_file_location:
            The location at which the sqlite_file_is_stored. The file is
            expected to have 3 tables: tanimoto_scores, inchikeys and
            spectra_data.
        s2v_model_file_name:
            File location of a spec2vec model. In addition two more files in
            the same folder are expected with the same name but with extensions
            .trainables.syn1neg.npy and .wv.vectors.npy.
        ms2ds_model_file_name:
            File location of a trained ms2ds model.
        pickled_s2v_embeddings_file_name:
            File location of a pickled file with Spec2Vec embeddings in a
            pd.Dataframe with as index the spectrum id.
        pickled_ms2ds_embeddings_file_name:
            File location of a pickled file with ms2ds embeddings in a
            pd.Dataframe with as index the spectrum id.

        **settings:
            As additional parameters predefined settings can be changed.
        spectrum_id_column_name:
            The name of the column or key in dictionaries under which the
            spectrum id is stored. Default = "spectrumid"
        cosine_score_tolerance:
            Setting for calculating the cosine score. If two peaks fall within
            the cosine_score tolerance the peaks are considered a match.
            Default = 0.1
        base_nr_mass_similarity:
            The base nr used for normalizing the mass similarity. Default = 0.8
        max_parent_mass:
            The value used to normalize the parent mass by dividing it by the
            max_parent_mass. Default = 13428.370894192036
        """
        # pylint: disable=too-many-arguments
        # Change default settings to values given in **settings
        settings = self._set_settings(settings)

        # Set given settings
        self.spectrum_id_column_name = settings["spectrum_id_column_name"]
        # todo create a ms2query model class that stores the model but also the
        #  settings used, since the settings used should always be the same as
        #  when the model was trained
        self.cosine_score_tolerance = settings["cosine_score_tolerance"]
        self.base_nr_mass_similarity = settings["base_nr_mass_similarity"]
        # todo make new model that has a fixed basic mass
        self.max_parent_mass = settings["max_parent_mass"]

        # Load models and set sqlite_file_location
        self.sqlite_file_location = sqlite_file_location
        self.s2v_model = Word2Vec.load(s2v_model_file_name)
        self.ms2ds_model = load_ms2ds_model(ms2ds_model_file_name)

        # loads the library embeddings into memory
        self.s2v_embeddings: pd.DataFrame = load_pickled_file(
            pickled_s2v_embeddings_file_name)
        self.ms2ds_embeddings: pd.DataFrame = load_pickled_file(
            pickled_ms2ds_embeddings_file_name)

    @staticmethod
    def _set_settings(new_settings: Dict[str, Union[str, bool]],
                      ) -> Dict[str, Union[str, float]]:
        """Changes default settings to new_settings and creates file names

        Args:
        ------
        new_settings:
            Dictionary with settings that should be changed. Only the
            keys given in default_settings can be used and the type has to be
            the same as the type of the values in default settings.
        """
        # Set default settings
        default_settings = {"spectrum_id_column_name": "spectrumid",
                            "cosine_score_tolerance": 0.1,
                            "base_nr_mass_similarity": 0.8,
                            "max_parent_mass": 13418.370894192036}

        for attribute in new_settings:
            assert attribute in default_settings, \
                f"Invalid argument in constructor:{attribute}"
            assert isinstance(new_settings[attribute],
                              type(default_settings[attribute])), \
                f"Different type is expected for argument: {attribute}"
            default_settings[attribute] = new_settings[attribute]
        return default_settings

    def select_best_matches(self,
                            query_spectra: List[Spectrum],
                            ms2query_model_file_name: str
                            ) -> Dict[str, pd.DataFrame]:
        """Returns ordered best matches with info for all query spectra

        Args
        ----
        query_spectra:
            List of query spectra for which the best matches should be found
        ms2query_model_file_name:
            File name of a hdf5 file containing the ms2query model.
        """
        # Selects top 20 best matches based on ms2ds and calculates all scores
        preselected_matches_info = \
            self.collect_matches_data_multiple_spectra(query_spectra)
        # Adds the ms2query model prediction to the dataframes
        preselected_matches_with_prediction = \
            self.get_ms2query_model_prediction(preselected_matches_info,
                                               ms2query_model_file_name)
        # todo decide for a good filtering method e.g. below certain treshold
        return preselected_matches_with_prediction

    def collect_matches_data_multiple_spectra(self,
                                              query_spectra: List[Spectrum],
                                              progress_bar=False
                                              ) -> Dict[str, pd.DataFrame]:
        """Returns a dataframe with info for all matches to all query spectra

        This is stored in a dictionary with as keys the spectrum_ids and as
        values a pd.Dataframe with on each row the information for one spectrum
        that was found in the preselection. The column names tell the info that
        is stored. Which column names/info is stored can be found in
        collect_data_for_tanimoto_prediction_model.

        Args:
        ------
        query_spectra:
            The spectra for which info about matches should be collected
        progress_bar:
            If true a progress bar is shown. Default is False
        """
        # Gets a preselection of spectra for all query_spectra
        dict_with_preselected_spectra = self.pre_select_spectra(query_spectra)

        # Run neural network model over all found spectra and add the predicted
        # scores to a dict with the spectra in dataframes
        dict_with_preselected_spectra_info = {}
        for query_spectrum in tqdm(query_spectra,
                                   desc="collecting matches info",
                                   disable=not progress_bar):
            spectrum_id = query_spectrum.get(self.spectrum_id_column_name)
            matches_with_info = \
                self._collect_data_for_ms2query_model(
                    query_spectrum,
                    dict_with_preselected_spectra[spectrum_id])
            dict_with_preselected_spectra_info[spectrum_id] = matches_with_info
        return dict_with_preselected_spectra_info

    def pre_select_spectra(self,
                           query_spectra: List[Spectrum],
                           nr_of_spectra: int = 20
                           ) -> Dict[str, List[str]]:
        """Returns dict with spectrum IDs of spectra with highest ms2ds score

        The keys are the query spectrum_ids and the values a list of the
        preselected spectrum_ids for each query spectrum.

        Args:
        ------
        query_spectra:
            spectra for which a preselection of possible library matches should
            be done
        nr_of_spectra:
            How many top spectra should be selected. Default = 20
        """
        dict_with_preselected_spectra = {}
        # Select top nr of spectra
        for query_spectrum in query_spectra:
            query_spectrum_id = query_spectrum.get(
                self.spectrum_id_column_name)
            ms2ds_scores = self._get_all_ms2ds_scores(query_spectrum)
            ms2ds_scores_np = ms2ds_scores["ms2ds_score"].to_numpy()
            # Get the indexes of the spectra with the highest ms2ds scores
            indexes_of_top_spectra = np.argpartition(
                ms2ds_scores_np,
                -nr_of_spectra,
                axis=0)[-nr_of_spectra:]
            # Select the spectra with the highest score
            selected_spectra = list(ms2ds_scores["ms2ds_score"].iloc[
                indexes_of_top_spectra].index)
            # Store selected spectra in dict
            dict_with_preselected_spectra[query_spectrum_id] = selected_spectra
        return dict_with_preselected_spectra

    def _get_all_ms2ds_scores(self, query_spectrum: Spectrum) -> pd.DataFrame:
        """Returns a dataframe with the ms2deepscore similarity scores

        query_spectrum
            Spectrum for which similarity scores should be calculated for all
            spectra in the ms2ds embeddings file.
        """
        ms2ds = MS2DeepScore(self.ms2ds_model, progress_bar=False)
        query_embedding = ms2ds.calculate_vectors([query_spectrum])
        library_ms2ds_embeddings_numpy = self.ms2ds_embeddings.to_numpy()

        ms2ds_scores = cosine_similarity_matrix(library_ms2ds_embeddings_numpy,
                                                query_embedding)
        similarity_matrix_dataframe = pd.DataFrame(
            ms2ds_scores,
            index=self.ms2ds_embeddings.index,
            columns=["ms2ds_score"])
        return similarity_matrix_dataframe

    def _collect_data_for_ms2query_model(
            self,
            query_spectrum: Spectrum,
            preselected_spectrum_ids: List[str]) -> pd.DataFrame:
        """Returns dataframe with relevant info for ms2query nn model

        query_spectrum:
            Spectrum for which all relevant data is collected
        preselected_spectrum_ids:
            List of spectrum ids that have the highest ms2ds scores with the
            query_spectrum
        """
        # pylint: disable=too-many-locals

        # Gets a list of all preselected spectra as Spectrum objects
        preselected_spectra_list = get_spectra_from_sqlite(
            self.sqlite_file_location,
            preselected_spectrum_ids,
            spectrum_id_storage_name=self.spectrum_id_column_name)
        # Gets cosine similarity matrix
        cosine_sim_matrix = CosineGreedy(
            tolerance=self.cosine_score_tolerance).matrix(
                preselected_spectra_list,
                [query_spectrum])
        # Gets modified cosine similarity matrix
        mod_cosine_sim_matrix = ModifiedCosine(
            tolerance=self.cosine_score_tolerance).matrix(
                preselected_spectra_list,
                [query_spectrum])
        # Changes [[(cos_score1, cos_match1)] [(cos_score2, cos_match2)]] into
        # [cos_score1, cos_score2], [cos_match1, cos_match2]
        cosine_score, cosine_matches = map(list, zip(
            *[x[0] for x in cosine_sim_matrix]))
        mod_cosine_score, mod_cosine_matches = map(list, zip(
            *[x[0] for x in mod_cosine_sim_matrix]))
        # Transform cosine score and mod cosine matches to in between 0-1
        normalized_cosine_matches = [1 - 0.93 ** i for i in cosine_matches]
        normalized_mod_cos_matches = \
            [1 - 0.93 ** i for i in mod_cosine_matches]

        parent_masses = [spectrum.get("parent_mass")
                         for spectrum in preselected_spectra_list]
        normalized_parent_masses = [parent_mass/self.max_parent_mass
                                    for parent_mass in parent_masses]

        mass_similarity = [self.base_nr_mass_similarity **
                           abs(spectrum.get("parent_mass") -
                               query_spectrum.get("parent_mass"))
                           for spectrum in preselected_spectra_list]

        # Get s2v_scores
        s2v_scores = Spec2Vec(self.s2v_model,
                              allowed_missing_percentage=100).matrix(
            create_spectrum_documents(preselected_spectra_list),
            create_spectrum_documents([query_spectrum]))[:, 0]

        # Get ms2ds_scores
        query_ms2ds_embeddings = MS2DeepScore(
            self.ms2ds_model,
            progress_bar=False).calculate_vectors([query_spectrum])
        preselected_ms2ds_embeddings = \
            self.ms2ds_embeddings.loc[preselected_spectrum_ids].to_numpy()
        ms2ds_scores = cosine_similarity_matrix(query_ms2ds_embeddings,
                                                preselected_ms2ds_embeddings
                                                )[0]

        # Add info together into a dataframe
        preselected_spectra_df = pd.DataFrame(
            {"cosine_score": cosine_score,
             "cosine_matches": normalized_cosine_matches,
             "mod_cosine_score": mod_cosine_score,
             "mod_cosine_matches": normalized_mod_cos_matches,
             "parent_mass": normalized_parent_masses,
             "mass_sim": mass_similarity,
             "s2v_scores": s2v_scores,
             "ms2ds_scores": ms2ds_scores},
            index=preselected_spectrum_ids)
        return preselected_spectra_df

    @staticmethod
    def get_ms2query_model_prediction(matches_info: Dict[str, pd.DataFrame],
                                      ms2query_model_file_name: str
                                      ) -> Dict[str, pd.DataFrame]:
        """Adds ms2query predictions to dataframes

        matches_info:
            A dictionary with as keys the query spectrum ids and as values
            pd.DataFrames containing the top 20 preselected matches and all
            info needed about these matches to run the ms2query model.
        ms2query_model_file_name:
            File name of a hdf5 name containing the ms2query model.
        """
        ms2query_nn_model = load_nn_model(ms2query_model_file_name)

        for query_spectrum_id in matches_info:
            current_query_matches_info = matches_info[query_spectrum_id]
            predictions = ms2query_nn_model.predict(current_query_matches_info)

            # Add prediction to dataframe
            current_query_matches_info[
                "ms2query_model_prediction"] = predictions
            matches_info[query_spectrum_id] = \
                current_query_matches_info.sort_values(
                    by=["ms2query_model_prediction"], ascending=False)
        return matches_info
