from typing import List, Dict, Union, Tuple, Set
import pandas as pd
import numpy as np
from tqdm import tqdm
from heapq import nlargest
from tensorflow.keras.models import load_model as load_nn_model
from gensim.models import Word2Vec
from matchms.Spectrum import Spectrum
from ms2deepscore.models import load_model as load_ms2ds_model
from ms2deepscore import MS2DeepScore
from spec2vec.vector_operations import cosine_similarity_matrix, calc_vector
from ms2query.query_from_sqlite_database import get_parent_mass_within_range, \
    get_parent_mass, get_inchikey_information
from ms2query.utils import load_pickled_file
from ms2query.spectrum_processing import create_spectrum_documents, \
    clean_metadata, minimal_processing_multiple_spectra
from ms2query.results_table import ResultsTable


class MS2Library:
    """Calculates scores of spectra in library and selects best matches

    For example

    .. code-block:: python

        from ms2query import MS2Library

        ms2library = MS2Library(sqlite_file_loc,
                                spec2vec_model_file_loc,
                                ms2ds_model_file_name,
                                s2v_pickled_embeddings_file,
                                ms2ds_embeddings_file_name,
                                spectrum_id_column_name)

    """
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
        progress_bars:
            If True progress bars will be shown of all methods. Default = True
        """
        # pylint: disable=too-many-arguments

        # Change default settings to values given in **settings
        self.settings = self._set_settings(settings)

        # Load models and set sqlite_file_location
        self.sqlite_file_location = sqlite_file_location
        self.s2v_model = Word2Vec.load(s2v_model_file_name)
        self.ms2ds_model = load_ms2ds_model(ms2ds_model_file_name)

        # loads the library embeddings into memory
        self.s2v_embeddings: pd.DataFrame = load_pickled_file(
            pickled_s2v_embeddings_file_name)
        self.ms2ds_embeddings: pd.DataFrame = load_pickled_file(
            pickled_ms2ds_embeddings_file_name)

        # load parent masses
        self.parent_masses_library = get_parent_mass(
            self.sqlite_file_location,
            self.settings["spectrum_id_column_name"])

        # Load inchikey information into memory
        self.spectra_of_inchikey14s, \
            self.closely_related_inchikey14s = \
            get_inchikey_information(self.sqlite_file_location)
        self.inchikey14s_of_spectra = {}
        for inchikey, list_of_spectrum_ids in \
                self.spectra_of_inchikey14s.items():
            for spectrum_id in list_of_spectrum_ids:
                self.inchikey14s_of_spectra[spectrum_id] = inchikey

    @staticmethod
    def _set_settings(new_settings: Dict[str, Union[str, bool]],
                      ) -> Dict[str, Union[str, float]]:
        """Changes default settings to new_settings

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
                            "progress_bars": True}
        for attribute in new_settings:
            assert attribute in default_settings, \
                f"Invalid argument in constructor:{attribute}"
            assert isinstance(new_settings[attribute],
                              type(default_settings[attribute])), \
                f"Different type is expected for argument: {attribute}"
            default_settings[attribute] = new_settings[attribute]
        return default_settings

    def analog_search(self,
                      query_spectra: List[Spectrum],
                      ms2query_model_file_name: str,
                      preselection_cut_off: int = 2000
                      ) -> List[ResultsTable]:
        """Returns a dictionary with a ResultTable for each query spectrum

        Args
        ----
        query_spectra:
            List of query spectra for which the best matches should be found
        ms2query_model_file_name:
            File name of a hdf5 file containing the ms2query model.
        preselection_cut_off:
            The number of spectra with the highest ms2ds that should be
            selected. Default = 2000
        """
        # TODO: remove ms2query_model_file_name from input parameters
        query_spectra = clean_metadata(query_spectra)
        query_spectra = minimal_processing_multiple_spectra(query_spectra)

        # Selects top 20 best matches based on ms2ds and calculates all scores
        analog_search_scores = \
            self._get_analog_search_scores(query_spectra, preselection_cut_off)
        # Adds the ms2query model prediction to the dataframes
        results_analog_search = \
            get_ms2query_model_prediction(analog_search_scores,
                                          ms2query_model_file_name)
        return results_analog_search

    def select_potential_true_matches(self,
                                      query_spectra: List[Spectrum],
                                      mass_tolerance: Union[float, int] = 0.1,
                                      s2v_score_threshold: float = 0.6
                                      ) -> List[pd.DataFrame]:
        """Returns potential true matches for query spectra

        The spectra are selected that fall within the mass_tolerance and have a
        s2v score higher than s2v_score_threshold.

        Args:
        ------
        query_spectra:
            A list with spectrum objects for which the potential true matches
            are returned
        mass_tolerance:
            The mass difference between query spectrum and library spectrum,
            that is allowed.
        s2v_score_threshold:
            The minimal s2v score to be considered a potential true match
        """
        query_spectra = clean_metadata(query_spectra)
        query_spectra = minimal_processing_multiple_spectra(query_spectra)

        found_matches_list = []
        for query_spectrum in tqdm(query_spectra,
                                   desc="Selecting potential perfect matches",
                                   disable=not self.settings["progress_bars"]):
            query_parent_mass = query_spectrum.get("parent_mass")
            # Preselection based on parent mass
            parent_masses_within_mass_tolerance = get_parent_mass_within_range(
                self.sqlite_file_location,
                query_parent_mass - mass_tolerance,
                query_parent_mass + mass_tolerance,
                self.settings["spectrum_id_column_name"])
            selected_library_spectra = [result[0] for result in
                                        parent_masses_within_mass_tolerance]
            s2v_scores = self._get_s2v_scores(query_spectrum,
                                              selected_library_spectra)
            found_matches = pd.DataFrame(columns=["spectrum_id", "s2v_score"])
            for i, spectrum_and_parent_mass in enumerate(parent_masses_within_mass_tolerance):
                if s2v_scores[i] > s2v_score_threshold:
                    found_matches = \
                        found_matches.append(
                            {"spectrum_id": spectrum_and_parent_mass[0],
                             "s2v_score": s2v_scores[i],
                             "parent_mass_difference": spectrum_and_parent_mass[1]},
                            ignore_index=True)
            found_matches.set_index("spectrum_id", inplace=True)
            found_matches_list.append(found_matches)
        return found_matches_list

    def _get_analog_search_scores(self,
                                  query_spectra: List[Spectrum],
                                  preselection_cut_off: int
                                  ) -> List[ResultsTable]:
        """Does preselection and returns scores for MS2Query model prediction

        This is stored in a dictionary with as keys the spectrum_ids and as
        values a pd.Dataframe with on each row the information for one spectrum
        that was found in the preselection. The column names tell the info that
        is stored. Which column names/info is stored can be found in
        collect_data_for_tanimoto_prediction_model.

        Args:
        ------
        query_spectra:
            The spectra for which info about matches should be collected
        preselection_cut_off:
            The number of spectra with the highest ms2ds that should be
            selected
        """
        ms2ds_scores = self._get_all_ms2ds_scores(query_spectra)

        result_tables = []
        for i, query_spectrum in \
                tqdm(enumerate(query_spectra),
                     desc="collecting matches info",
                     disable=not self.settings["progress_bars"]):

            results_table = ResultsTable(
                preselection_cut_off=preselection_cut_off,
                ms2deepscores=ms2ds_scores.iloc[:, i],
                query_spectrum=query_spectrum,
                sqlite_file_name=self.sqlite_file_location)

            # Select the library spectra that have the highest MS2Deepscore
            results_table.preselect_on_ms2deepscore()
            # Calculate the average ms2ds scores and neigbourhood score
            results_table = \
                self._calculate_averages_and_chemical_neigbhourhood_score(
                    results_table)
            results_table.data = results_table.data.set_index('spectrum_ids')

            results_table.data["s2v_score"] = self._get_s2v_scores(
                query_spectrum,
                results_table.data.index.values)

            parent_masses = np.array(
                [self.parent_masses_library[x]
                 for x in results_table.data.index])
            results_table.add_parent_masses(
                parent_masses,
                self.settings["base_nr_mass_similarity"])

            result_tables.append(results_table)

        return result_tables

    def _get_all_ms2ds_scores(self, query_spectra: List[Spectrum]
                              ) -> pd.DataFrame:
        """Returns a dataframe with the ms2deepscore similarity scores

        The similarity scores are calculated between the query_spectra and all
        library spectra.

        query_spectra
            Spectra for which similarity scores should be calculated for all
            spectra in the ms2ds embeddings file.
        """
        ms2ds = MS2DeepScore(self.ms2ds_model, progress_bar=False)
        if self.settings["progress_bars"]:
            print("Calculating MS2Deepscore embeddings for query spectra")
        query_embeddings = ms2ds.calculate_vectors(query_spectra)
        library_ms2ds_embeddings_numpy = self.ms2ds_embeddings.to_numpy()
        if self.settings["progress_bars"]:
            print("Calculating MS2Deepscore between query embeddings and "
                  "library embeddings")
        ms2ds_scores = cosine_similarity_matrix(library_ms2ds_embeddings_numpy,
                                                query_embeddings)
        similarity_matrix_dataframe = pd.DataFrame(
            ms2ds_scores,
            index=self.ms2ds_embeddings.index)
        return similarity_matrix_dataframe

    def _calculate_averages_and_chemical_neigbhourhood_score(
            self,
            results_table: ResultsTable
            ):
        """Returns preselected spectra and average and closely related ms2ds

        results_table:
            ResultsTable object to collect scores and data about spectra of interest.
        """
        selected_spectrum_ids = list(results_table.data["spectrum_ids"])
        ms2ds_scores = results_table.ms2deepscores
        selected_inchikeys = \
            [self.inchikey14s_of_spectra[x] for x in selected_spectrum_ids]
        # Populate results table
        results_table.data["inchikey"] = selected_inchikeys
        selected_inchikeys_set = set(selected_inchikeys)

        # Select inchikeys for which the average ms2ds scores should be
        # calculated
        selected_closely_related_inchikeys = []
        for inchikey in selected_inchikeys_set:
            selected_closely_related_inchikeys += \
                [scores[0] for scores in self.closely_related_inchikey14s[inchikey]]
        inchikeys_to_calc_average_for = \
            set(selected_closely_related_inchikeys) | selected_inchikeys_set

        average_ms2ds_scores = \
            self._get_average_ms2ds_for_inchikey14(
                ms2ds_scores, inchikeys_to_calc_average_for)

        closely_related_inchikey_scores = self._get_chemical_neighbourhood_scores(
            selected_inchikeys_set,
            average_ms2ds_scores)
        results_table.add_related_inchikey_scores(closely_related_inchikey_scores)
        results_table.add_average_ms2ds_scores(average_ms2ds_scores)
        return results_table

    def _get_s2v_scores(self,
                        query_spectrum: Spectrum,
                        preselection_of_library_ids: List[str]
                        ) -> np.ndarray:
        """Returns the s2v scores

        query_spectrum:
            Spectrum object
        preselection_of_library_ids:
            list of spectrum ids for which the s2v scores should be calculated
            """
        query_spectrum_document = \
            create_spectrum_documents([query_spectrum])[0]
        query_s2v_embedding = calc_vector(self.s2v_model,
                                          query_spectrum_document,
                                          allowed_missing_percentage=100)
        preselected_s2v_embeddings = \
            self.s2v_embeddings.loc[preselection_of_library_ids].to_numpy()
        s2v_scores = cosine_similarity_matrix(np.array([query_s2v_embedding]),
                                              preselected_s2v_embeddings)[0]
        # todo convert to dataframe, so there is less chance of introducing
        #  errors
        return s2v_scores

    def _get_average_ms2ds_for_inchikey14(self,
                                          ms2ds_scores: pd.DataFrame,
                                          inchikey14s: Set[str]
                                          ) -> Dict[str, Tuple[float, int]]:
        """Returns the average ms2ds score per inchikey

        Args:
        ------
        ms2ds_scores:
            The ms2ds scores with as index the library spectrum ids and as
            values the ms2ds scores.
        inchikey14s:
            Set of inchikeys to average over.
        """
        inchikey14_scores = {}
        for inchikey14 in inchikey14s:
            sum_of_ms2ds_scores = 0
            for spectrum_id in self.spectra_of_inchikey14s[inchikey14]:
                sum_of_ms2ds_scores += ms2ds_scores.loc[spectrum_id]
            nr_of_spectra = len(self.spectra_of_inchikey14s[inchikey14])
            if nr_of_spectra > 0:
                avg_ms2ds_score = sum_of_ms2ds_scores / nr_of_spectra
                inchikey14_scores[inchikey14] = (avg_ms2ds_score,
                                                 nr_of_spectra)
        return inchikey14_scores

    def _preselect_best_matching_inchikeys(
            self,
            average_ms2ds_scores_per_inchikey14: Dict[str, Tuple[float, int]],
            nr_of_top_inchikey14s_to_select: int
            ) -> Tuple[List[str], List[str]]:
        """Does a preselection based on the average_ms2ds_scores

        Returns the inchikeys with the highest average ms2ds score and returns
        all spectra belonging to these inchikeys

        Args:
        ------
        average_ms2ds_scores_per_inchikey14:
            The average ms2ds scores per inchikey, the keys of the dictionary
            are the inchikeys and the values are a tuple containing the
            average ms2ds score and the number of spectra belonging to this
            inchikey.
        nr_of_top_inchikey14s_to_select:
            The number of inchikeys that are selected
        """
        # is not used in the current workflow but might be added again later
        # Selects the inchikeys with the highest average ms2ds scores
        top_inchikeys = nlargest(nr_of_top_inchikey14s_to_select,
                                 average_ms2ds_scores_per_inchikey14,
                                 key=average_ms2ds_scores_per_inchikey14.get)
        top_spectrum_ids = []
        for inchikey in top_inchikeys:
            top_spectrum_ids += self.spectra_of_inchikey14s[inchikey]

        return top_inchikeys, top_spectrum_ids

    def _get_chemical_neighbourhood_scores(
            self,
            selected_inchikey14s: Set[str],
            average_inchikey_scores: Dict[str, Tuple[float, int]]
            ) -> Dict[str, Tuple[float, int, float]]:
        """Returns the chemical neighbourhood scores for selected inchikey14s

        A dictionary is returned with as keys the inchikeys and als value a
        tuple with the chemical neighbourhood score, the number of spectra
        used for this score and the weight used for this score.
        The closely related score is calculated by taking the average inchikey
        score times the nr of spectra with this inchikey times the tanimoto
        score of this spectrum devided by the total weight of all scores
        combined.

        Args:
        ------
        selected_inchikey14s:
            The inchikeys for which the chemical neighbourhood scores are
            calculated
        average_inchikey_scores:
            Dictionary containing the average MS2Deepscore scores for each
            inchikey and the number of spectra belonging to this inchikey.
        """
        related_inchikey_score_dict = {}
        for inchikey in selected_inchikey14s:
            # For each inchikey a list with the top 10 closest related inchikeys
            #  and the corresponding tanimoto score is stored
            best_matches_and_tanimoto_scores = \
                self.closely_related_inchikey14s[inchikey]

            # Count the weight, nr and sum of tanimoto scores to calculate the
            #  average tanimoto score.
            sum_related_inchikey_tanimoto_scores = 0
            total_weight_of_spectra_used = 0
            total_nr_of_spectra_used = 0
            for closely_related_inchikey14, tanimoto_score in \
                    best_matches_and_tanimoto_scores:
                # Get the ms2ds score for this closely related inchikey14 and the
                # nr of spectra for this related inchikey.
                closely_related_ms2ds, nr_of_spectra_related_inchikey14 = \
                    average_inchikey_scores[closely_related_inchikey14]
                # todo think of different weighting based on tanimoto score,
                #  e.g. nr_of_spectra^tanimoto_score or return all individual
                #  scores, nr and tanimoto score for each closely related inchikey
                #  (so 30 in total) to MS2Query
                # The weight of closely related spectra is based on the tanimoto
                # score and the nr of spectra this inchikey has.
                weight_of_closely_related_inchikey_score = \
                    nr_of_spectra_related_inchikey14 * tanimoto_score

                sum_related_inchikey_tanimoto_scores += \
                    closely_related_ms2ds * \
                    weight_of_closely_related_inchikey_score
                total_weight_of_spectra_used += \
                    weight_of_closely_related_inchikey_score
                total_nr_of_spectra_used += nr_of_spectra_related_inchikey14

            average_tanimoto_score_used = \
                total_weight_of_spectra_used/total_nr_of_spectra_used

            related_inchikey_score_dict[
                inchikey] = \
                (sum_related_inchikey_tanimoto_scores/total_weight_of_spectra_used,
                 total_nr_of_spectra_used,
                 average_tanimoto_score_used)
        return related_inchikey_score_dict


def get_ms2query_model_prediction(
        matches_info: List[Union[ResultsTable, None]],
        ms2query_model_file_name: str
        ) -> List[ResultsTable]:
    """Adds ms2query predictions to dataframes

    matches_info:
        A dictionary with as keys the query spectrum ids and as values
        pd.DataFrames containing the top 20 preselected matches and all
        info needed about these matches to run the ms2query model.
    ms2query_model_file_name:
        File name of a hdf5 name containing the ms2query model.
    """
    ms2query_nn_model = load_nn_model(ms2query_model_file_name)
    for result_table in matches_info:
        current_query_matches_info = result_table.get_training_data().copy()
        predictions = ms2query_nn_model.predict(current_query_matches_info)

        result_table.add_ms2query_meta_score(predictions)

    return matches_info
