import os.path
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
    get_parent_mass, get_inchikey_information, get_metadata_from_sqlite
from ms2query.utils import load_pickled_file, get_classifier_from_csv_file
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
                 sqlite_file_name: str,
                 s2v_model_file_name: str,
                 ms2ds_model_file_name: str,
                 pickled_s2v_embeddings_file_name: str,
                 pickled_ms2ds_embeddings_file_name: str,
                 ms2query_model_file_name: Union[str, None],
                 classifier_csv_file_name: Union[str, None] = None,
                 **settings):
        """
        Parameters
        ----------
        sqlite_file_name:
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
        ms2query_model_file_name:
            File location of ms2query model with .hdf5 extension.
        classifier_csv_file_name:
            Csv file location containing classifier annotations per inchikey

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

        # Load models and set file locations
        self.classifier_file_name = classifier_csv_file_name
        assert os.path.isfile(sqlite_file_name), f"The given sqlite file does not exist: {sqlite_file_name}"
        self.sqlite_file_name = sqlite_file_name
        if ms2query_model_file_name is not None:
            self.ms2query_model = load_nn_model(ms2query_model_file_name)
        else:
            self.ms2query_model = None

        self.s2v_model = Word2Vec.load(s2v_model_file_name)
        self.ms2ds_model = load_ms2ds_model(ms2ds_model_file_name)

        # loads the library embeddings into memory
        self.s2v_embeddings: pd.DataFrame = load_pickled_file(
            pickled_s2v_embeddings_file_name)
        self.ms2ds_embeddings: pd.DataFrame = load_pickled_file(
            pickled_ms2ds_embeddings_file_name)

        # load parent masses
        self.parent_masses_library = get_parent_mass(
            self.sqlite_file_name,
            self.settings["spectrum_id_column_name"])

        # Load inchikey information into memory
        self.spectra_of_inchikey14s, \
            self.closely_related_inchikey14s = \
            get_inchikey_information(self.sqlite_file_name)
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

    def analog_search_return_results_tables(self,
                                            query_spectra: List[Spectrum],
                                            preselection_cut_off: int = 2000
                                            ) -> List[ResultsTable]:
        """Returns a list with a ResultTable for each query spectrum

        Args
        ----
        query_spectra:
            List of query spectra for which the best matches should be found
        preselection_cut_off:
            The number of spectra with the highest ms2ds that should be
            selected. Default = 2000
        """
        assert self.ms2query_model is not None, \
            "MS2Query model should be given when creating ms2library object"
        query_spectra = clean_metadata(query_spectra)
        query_spectra = minimal_processing_multiple_spectra(query_spectra)

        # Calculate all ms2ds scores between all query and library spectra
        all_ms2ds_scores = self._get_all_ms2ds_scores(query_spectra)

        result_tables = []
        for i, query_spectrum in \
                tqdm(enumerate(query_spectra),
                     desc="collecting matches info",
                     disable=not self.settings["progress_bars"]):
            # Initialize result table
            results_table = ResultsTable(
                preselection_cut_off=preselection_cut_off,
                ms2deepscores=all_ms2ds_scores.iloc[:, i],
                query_spectrum=query_spectrum,
                sqlite_file_name=self.sqlite_file_name,
                classifier_csv_file_name=self.classifier_file_name)
            results_table = \
                self._calculate_scores_for_metascore(results_table)
            results_table = get_ms2query_model_prediction_single_spectrum(results_table, self.ms2query_model)
            result_tables.append(results_table)
        return result_tables

    def analog_search_store_in_csv(self,
                                   query_spectra: List[Spectrum],
                                   results_csv_file_location: str,
                                   preselection_cut_off: int = 2000,
                                   nr_of_top_analogs_to_save: int = 1,
                                   minimal_ms2query_metascore: Union[float, int] = 0.0
                                   ) -> None:
        """Stores the results of an analog in csv files.

        This method is less memory intensive than analog_search_return_results_table,
        since the results tables do not have to be kept in memory, since they are directly
        stored in a csv file.

        Args
        ----
        query_spectra:
            List of query spectra for which the best matches should be found
        results_csv_file_location:
            file location were a csv file is created that stores the results
        preselection_cut_off:
            The number of spectra with the highest ms2ds that should be
            selected. Default = 2000
        nr_of_top_analogs_to_save:
            The number of returned analogs that are stored.
        minimal_ms2query_metascore:
            The minimal ms2query metascore needed to be stored in the csv file.
            Spectra for which no analog with this minimal metascore was found,
            will not be stored in the csv file.
        """
        # pylint: disable=too-many-arguments

        # Create csv file if it does not exist already
        assert not os.path.exists(results_csv_file_location), "Csv file location for results already exists"
        assert self.ms2query_model is not None, \
            "MS2Query model should be given when creating ms2library object"

        with open(results_csv_file_location, "w", encoding="utf-8") as csv_file:
            if self.classifier_file_name is None:
                csv_file.write(",parent_mass_query_spectrum,ms2query_model_prediction,parent_mass_analog,inchikey,"
                               "spectrum_ids,analog_compound_name\n")
            else:
                csv_file.write(",parent_mass_query_spectrum,ms2query_model_prediction,parent_mass_analog,inchikey,"
                               "spectrum_ids,analog_compound_name,smiles,cf_kingdom,cf_superclass,cf_class,cf_subclass,"
                               "cf_direct_parent,npc_class_results,npc_superclass_results,npc_pathway_results\n")
        # preprocess spectra
        query_spectra = clean_metadata(query_spectra)
        query_spectra = minimal_processing_multiple_spectra(query_spectra)

        # Calculate all ms2ds scores between all query and library spectra
        all_ms2ds_scores = self._get_all_ms2ds_scores(query_spectra)

        for i, query_spectrum in \
                tqdm(enumerate(query_spectra),
                     desc="collecting matches info",
                     disable=not self.settings["progress_bars"]):
            # Initialize result table
            results_table = ResultsTable(
                preselection_cut_off=preselection_cut_off,
                ms2deepscores=all_ms2ds_scores.iloc[:, i],
                query_spectrum=query_spectrum,
                sqlite_file_name=self.sqlite_file_name,
                classifier_csv_file_name=self.classifier_file_name)
            results_table = \
                self._calculate_scores_for_metascore(results_table)
            results_table = get_ms2query_model_prediction_single_spectrum(results_table, self.ms2query_model)
            results_df = results_table.export_to_dataframe(nr_of_top_analogs_to_save, minimal_ms2query_metascore)
            if results_df is not None:
                results_df.to_csv(results_csv_file_location, mode="a", header=False)

    def select_potential_true_matches(self,
                                      query_spectra: List[Spectrum],
                                      mass_tolerance: Union[float, int] = 0.1,
                                      s2v_score_threshold: float = 0.6
                                      ) -> pd.DataFrame:
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

        found_matches = pd.DataFrame(columns=["query_spectrum_nr",
                                              "query_spectrum_parent_mass",
                                              "s2v_score",
                                              "match_spectrum_id",
                                              "match_parent_mass",
                                              "match_inchikey"])
        for query_spectrum_nr, query_spectrum in tqdm(enumerate(query_spectra),
                                                      desc="Selecting potential perfect matches",
                                                      disable=not self.settings["progress_bars"]):
            query_parent_mass = query_spectrum.get("parent_mass")
            # Preselection based on parent mass
            parent_masses_within_mass_tolerance = get_parent_mass_within_range(
                self.sqlite_file_name,
                query_parent_mass - mass_tolerance,
                query_parent_mass + mass_tolerance,
                self.settings["spectrum_id_column_name"])
            selected_library_spectra = [result[0] for result in
                                        parent_masses_within_mass_tolerance]
            s2v_scores = self._get_s2v_scores(query_spectrum,
                                              selected_library_spectra)

            for i, spectrum_id_and_parent_mass in enumerate(parent_masses_within_mass_tolerance):
                match_spectrum_id, match_parent_mass = spectrum_id_and_parent_mass
                if s2v_scores[i] > s2v_score_threshold:
                    found_matches = \
                        found_matches.append(
                            {"query_spectrum_nr": query_spectrum_nr,
                             "query_spectrum_parent_mass": query_parent_mass,
                             "s2v_score": s2v_scores[i],
                             "match_spectrum_id": match_spectrum_id,
                             "match_parent_mass": match_parent_mass,
                             "match_inchikey": self.inchikey14s_of_spectra[match_spectrum_id]},
                            ignore_index=True)
        return found_matches

    def store_potential_true_matches(self,
                                     query_spectra: List[Spectrum],
                                     results_file_location: str,
                                     mass_tolerance: Union[float, int] = 0.1,
                                     s2v_score_threshold: float = 0.6
                                     ) -> None:
        """Stores the results of a library search in a csv file

        The spectra are selected that fall within the mass_tolerance and have a
        s2v score higher than s2v_score_threshold.

        Args:
        ------
        query_spectra:
            A list with spectrum objects for which the potential true matches
            are returned
        results_file_location:
            File location in which a csv file is created containing the results
        mass_tolerance:
            The mass difference between query spectrum and library spectrum,
            that is allowed.
        s2v_score_threshold:
            The minimal s2v score to be considered a potential true match
        """
        assert not os.path.exists(results_file_location), "Results file already exists"
        found_matches = self.select_potential_true_matches(query_spectra,
                                                           mass_tolerance,
                                                           s2v_score_threshold)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        # For each analog the compound name is selected from sqlite
        metadata_dict = get_metadata_from_sqlite(self.sqlite_file_name,
                                                 list(found_matches["match_spectrum_id"]))
        compound_name_list = [metadata_dict[match_spectrum_id]["compound_name"]
                              for match_spectrum_id
                              in list(found_matches["match_spectrum_id"])]
        found_matches["match_compound_name"] = compound_name_list
        if self.classifier_file_name is not None and not found_matches.empty:
            classifier_data = get_classifier_from_csv_file(self.classifier_file_name,
                                                           list(found_matches["match_inchikey"].unique()))
            classifier_data.rename(columns={"inchikey": "match_inchikey"}, inplace=True)
            found_matches = found_matches.merge(classifier_data, on="match_inchikey")
        found_matches.to_csv(results_file_location, mode="w", index=False)

    def _calculate_scores_for_metascore(self,
                                        results_table: ResultsTable
                                        ) -> ResultsTable:
        """Calculate the needed scores for metascore for selected spectra

        Args:
        ------
        results_table:
            ResultsTable object for which no scores have been selected yet.
        """
        # Select the library spectra that have the highest MS2Deepscore
        results_table.preselect_on_ms2deepscore()
        # Calculate the average ms2ds scores and neigbourhood score
        results_table = \
            self._calculate_averages_and_chemical_neigbhourhood_score(
                results_table)
        results_table.data = results_table.data.set_index('spectrum_ids')

        results_table.data["s2v_score"] = self._get_s2v_scores(
            results_table.query_spectrum,
            results_table.data.index.values)

        parent_masses = np.array(
            [self.parent_masses_library[x]
             for x in results_table.data.index])
        results_table.add_parent_masses(
            parent_masses,
            self.settings["base_nr_mass_similarity"])
        return results_table

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


def get_ms2query_model_prediction_single_spectrum(
        result_table: Union[ResultsTable, None],
        ms2query_nn_model
        ) -> ResultsTable:
    """Adds ms2query predictions to result table

    result_table:
    ms2query_model_file_name:
        File name of a hdf5 name containing the ms2query model.
    """
    current_query_matches_info = result_table.get_training_data().copy()
    predictions = ms2query_nn_model.predict(current_query_matches_info)

    result_table.add_ms2query_meta_score(predictions)

    return result_table


def create_library_object_from_one_dir(directory: str,
                                       file_name_dictionary: Dict[str, str]
                                       ) -> MS2Library:
    """Creates a library object for specified directory and file names

    For default file names the function run_ms2query.default_library_file_names can be used

    Args:
    ------
    directory:
        Path to the directory in which the files are stored
    file_name_dictionary:
        A dictionary with as keys the type of file and as values the base names of the files
    """
    sqlite_file_name = os.path.join(directory, file_name_dictionary["sqlite"])
    if file_name_dictionary["classifiers"] is not None:
        classifiers_file_name = os.path.join(directory, file_name_dictionary["classifiers"])
    else:
        classifiers_file_name = None

    # Models
    s2v_model_file_name = os.path.join(directory, file_name_dictionary["s2v_model"])
    ms2ds_model_file_name = os.path.join(directory, file_name_dictionary["ms2ds_model"])
    ms2query_model_file_name = os.path.join(directory, file_name_dictionary["ms2query_model"])

    # Embeddings
    s2v_embeddings_file_name = os.path.join(directory, file_name_dictionary["s2v_embeddings"])
    ms2ds_embeddings_file_name = os.path.join(directory, file_name_dictionary["ms2ds_embeddings"])

    return MS2Library(sqlite_file_name, s2v_model_file_name, ms2ds_model_file_name, s2v_embeddings_file_name,
                      ms2ds_embeddings_file_name, ms2query_model_file_name, classifiers_file_name)
