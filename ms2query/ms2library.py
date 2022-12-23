import os.path
from typing import Dict, List, Set, Tuple, Union, Optional
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from matchms.Spectrum import Spectrum
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model as load_ms2ds_model
from spec2vec.vector_operations import calc_vector, cosine_similarity_matrix
from tqdm import tqdm
from ms2query.query_from_sqlite_database import (get_inchikey_information,
                                                 get_precursor_mz, get_ionization_mode_library)
from ms2query.results_table import ResultsTable
from ms2query.clean_and_filter_spectra import (clean_metadata,
                                               create_spectrum_documents,
                                               normalize_and_filter_peaks)
from ms2query.utils import (column_names_for_output, load_ms2query_model,
                            load_pickled_file, SettingsRunMS2Query)


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
            self.ms2query_model = load_ms2query_model(ms2query_model_file_name)

        self.s2v_model = Word2Vec.load(s2v_model_file_name)
        self.ms2ds_model = load_ms2ds_model(ms2ds_model_file_name)

        # loads the library embeddings into memory
        self.s2v_embeddings: pd.DataFrame = load_pickled_file(
            pickled_s2v_embeddings_file_name)
        self.ms2ds_embeddings: pd.DataFrame = load_pickled_file(
            pickled_ms2ds_embeddings_file_name)
        
        assert self.ms2ds_model.base.output_shape[1] == self.ms2ds_embeddings.shape[1], \
            "Dimension of pre-computed MS2DeepScore embeddings does not fit given model."

        # load precursor mz's
        self.precursors_library = get_precursor_mz(
            self.sqlite_file_name)

        assert self.ms2ds_embeddings.shape[0] == self.s2v_embeddings.shape[0], \
            "The number ms2deepscore embeddings is not equal to the number of spectra with s2v embeddings"

        assert self.ms2ds_embeddings.shape[0] == len(self.precursors_library), \
            "Mismatch of library files. " \
            "The number of spectra in the sqlite library is not equal to the number of spectra in the embeddings"

        self.ionization_mode = get_ionization_mode_library(self.sqlite_file_name)
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
                            "progress_bars": True}
        for attribute in new_settings:
            assert attribute in default_settings, \
                f"Invalid argument in constructor:{attribute}"
            assert isinstance(new_settings[attribute],
                              type(default_settings[attribute])), \
                f"Different type is expected for argument: {attribute}"
            default_settings[attribute] = new_settings[attribute]
        return default_settings

    def calculate_features_single_spectrum(self,
                                           query_spectrum: Spectrum,
                                           preselection_cut_off: int = 2000,
                                           filter_on_ionmode: Optional[str] = None) -> Optional[ResultsTable]:
        """Calculates a results table for a single spectrum"""
        query_spectrum = clean_metadata(query_spectrum)
        query_spectrum = normalize_and_filter_peaks(query_spectrum)
        if query_spectrum is None:
            return None

        # Check if the ionization mode matches that of the library
        query_ionmode = query_spectrum.get("ionmode")
        if filter_on_ionmode is not None:
            if query_ionmode != filter_on_ionmode:
                print(f"This spectrum is not analyzed since it was not in {filter_on_ionmode} ionization mode. "
                      f"Instead the spectrum is in {query_ionmode} ionization mode.")
                return None
        if query_ionmode != "n/a" and self.ionization_mode is not None:
            assert query_ionmode == self.ionization_mode, \
                f"The spectrum is in {query_ionmode} ionization mode, while the library is for {self.ionization_mode} ionization mode. " \
                f"Check the readme to download a library in the {query_ionmode} ionization mode"


        ms2deepscore_scores = self._get_all_ms2ds_scores(query_spectrum)
        # Initialize result table
        results_table = ResultsTable(
            preselection_cut_off=preselection_cut_off,
            ms2deepscores=ms2deepscore_scores,
            query_spectrum=query_spectrum,
            sqlite_file_name=self.sqlite_file_name,
            classifier_csv_file_name=self.classifier_file_name)
        results_table = \
            self._calculate_features_for_random_forest_model(results_table)
        return results_table

    def analog_search_return_results_tables(self,
                                            query_spectra: List[Spectrum],
                                            preselection_cut_off: int = 2000,
                                            store_ms2deepscore_scores: bool = False
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

        result_tables = []
        for i, query_spectrum in tqdm(enumerate(query_spectra),
                                      desc="collecting matches info",
                                      disable=not self.settings["progress_bars"],
                                      total=len(query_spectra)):
            query_spectrum.set("spectrum_nr", i+1)
            results_table = self.calculate_features_single_spectrum(query_spectrum, preselection_cut_off)
            if results_table is not None:
                results_table = get_ms2query_model_prediction_single_spectrum(results_table, self.ms2query_model)
                # To reduce the memory footprint the ms2deepscore scores are removed.
                if not store_ms2deepscore_scores:
                    results_table.ms2deepscores = pd.DataFrame()
                result_tables.append(results_table)
            else:
                result_tables.append(None)
        return result_tables

    def analog_search_store_in_csv(self,
                                   query_spectra: List[Spectrum],
                                   results_csv_file_location: str,
                                   settings: Optional[SettingsRunMS2Query] = None
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
        settings:
            Settings for running MS2Query, see SettingsRunMS2Query for details.
        """
        if settings is None:
            settings = SettingsRunMS2Query()
        # Create csv file if it does not exist already
        assert not os.path.exists(results_csv_file_location), "Csv file location for results already exists"
        assert self.ms2query_model is not None, \
            "MS2Query model should be given when creating ms2library object"

        with open(results_csv_file_location, "w", encoding="utf-8") as csv_file:
            if self.classifier_file_name is None:
                csv_file.write(",".join(column_names_for_output(True, False, settings.additional_metadata_columns,
                                                                settings.additional_ms2query_score_columns)) + "\n")
            else:
                csv_file.write(",".join(column_names_for_output(True, True, settings.additional_metadata_columns,
                                                                settings.additional_ms2query_score_columns)) + "\n")

        for i, query_spectrum in \
                tqdm(enumerate(query_spectra),
                     desc="Predicting matches for query spectra",
                     disable=not self.settings["progress_bars"],
                     total=len(query_spectra)):
            query_spectrum.set("spectrum_nr", i+1)
            results_table = self.calculate_features_single_spectrum(query_spectrum, settings.preselection_cut_off,
                                                                    settings.filter_on_ion_mode)
            if results_table is None:
                print(f"Spectrum nr {i} was not stored, since it did not pass all cleaning steps")
            else:
                results_table = get_ms2query_model_prediction_single_spectrum(results_table, self.ms2query_model)
                results_df = results_table.export_to_dataframe(
                    settings.nr_of_top_analogs_to_save,
                    settings.minimal_ms2query_metascore,
                    additional_metadata_columns=settings.additional_metadata_columns,
                    additional_ms2query_score_columns=settings.additional_ms2query_score_columns)
                results_df.to_csv(results_csv_file_location, mode="a", header=False, float_format="%.4f", index=False)

    def _calculate_features_for_random_forest_model(self,
                                                    results_table: ResultsTable
                                                    ) -> ResultsTable:
        """Calculate the features for random forest model for selected spectra

        Args:
        ------
        results_table:
            ResultsTable object for which no scores have been selected yet.
        """
        # Select the library spectra that have the highest MS2Deepscore
        results_table.preselect_on_ms2deepscore()
        # Calculate the average ms2ds scores and neigbourhood score
        results_table = \
            self._calculate_average_ms2deepscore_multiple_library_spectra(results_table)
        results_table.data = results_table.data.set_index('spectrum_ids')

        results_table.data["s2v_score"] = self._get_s2v_scores(
            results_table.query_spectrum,
            results_table.data.index.values)

        precursors = np.array(
            [self.precursors_library[x]
             for x in results_table.data.index])
        results_table.add_precursors(
            precursors)
        return results_table

    def _get_all_ms2ds_scores(self, query_spectrum: Spectrum
                              ) -> pd.Series:
        """Returns a dataframe with the ms2deepscore similarity scores

        The similarity scores are calculated between the query_spectrum and all
        library spectra.

        query_spectra
            Spectrum for which similarity scores should be calculated for all
            spectra in the ms2ds embeddings file.
        """
        ms2ds = MS2DeepScore(self.ms2ds_model, progress_bar=False)
        query_embeddings = ms2ds.calculate_vectors([query_spectrum])
        library_ms2ds_embeddings_numpy = self.ms2ds_embeddings.to_numpy()
        ms2ds_scores = cosine_similarity_matrix(library_ms2ds_embeddings_numpy,
                                                query_embeddings)
        similarity_matrix_dataframe = pd.DataFrame(
            ms2ds_scores,
            index=self.ms2ds_embeddings.index)
        return similarity_matrix_dataframe.iloc[:, 0]

    def _calculate_average_ms2deepscore_multiple_library_spectra(
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

        # Select inchikeys for which the average ms2ds scores should be calculated
        selected_closely_related_inchikeys = []
        for inchikey in selected_inchikeys_set:
            selected_closely_related_inchikeys += \
                [scores[0] for scores in self.closely_related_inchikey14s[inchikey]]

        inchikeys_to_calc_average_for = \
            set(selected_closely_related_inchikeys)

        average_ms2ds_scores_per_inchikey = \
            self._get_average_ms2ds_for_inchikey14(
                ms2ds_scores, inchikeys_to_calc_average_for)
        closely_related_inchikey_scores = \
            self._calculate_average_multiple_library_structures(selected_inchikeys_set,
                                                                average_ms2ds_scores_per_inchikey)

        results_table.add_related_inchikey_scores(closely_related_inchikey_scores)
        return results_table

    def _get_average_ms2ds_for_inchikey14(self,
                                          ms2ds_scores: pd.DataFrame,
                                          inchikey14s: Set[str]
                                          ) -> Dict[str, float]:
        """Returns the average ms2ds score per inchikey

        Args:
        ------
        ms2ds_scores:
            The ms2ds scores with as index the library spectrum ids and as
            values the ms2ds scores.
        inchikey14s:
            Set of inchikeys to average over.
        """
        average_ms2ds_per_inchikey14 = {}
        for inchikey14 in inchikey14s:
            sum_of_ms2ds_scores = 0
            for spectrum_id in self.spectra_of_inchikey14s[inchikey14]:
                sum_of_ms2ds_scores += ms2ds_scores.loc[spectrum_id]
            nr_of_spectra = len(self.spectra_of_inchikey14s[inchikey14])
            if nr_of_spectra > 0:
                avg_ms2ds_score = sum_of_ms2ds_scores / nr_of_spectra
                average_ms2ds_per_inchikey14[inchikey14] = avg_ms2ds_score
        return average_ms2ds_per_inchikey14

    def _calculate_average_multiple_library_structures(
            self,
            selected_inchikey14s: Set[str],
            average_inchikey_scores: Dict[str, float]
            ) -> Dict[str, Tuple[float, float]]:
        """Returns the average ms2deepscore and average tanimoto score for the 10 chemically closest inchikeys

        Args:
        ------
        selected_inchikey14s:
            The inchikeys for which the 10 chemically closest inchikeys are selected
        average_inchikey_scores:
            Dictionary containing the average MS2Deepscore scores for each
            inchikey and the number of spectra belonging to this inchikey.
        """
        results_per_inchikey = {}
        for inchikey in selected_inchikey14s:
            # For each inchikey a list with the top 10 closest related inchikeys
            #  and the corresponding tanimoto score is stored
            closest_library_structures_and_tanimoto_scores = \
                self.closely_related_inchikey14s[inchikey]
            closest_library_structures = [tuple[0] for tuple in closest_library_structures_and_tanimoto_scores]
            tanimoto_scores = [tuple[1] for tuple in closest_library_structures_and_tanimoto_scores]

            average_tanimoto_score_multiple_library_spectra = sum(tanimoto_scores) / len(tanimoto_scores)
            sum_of_average_ms2ds_multiple_library_structures = \
                sum(average_inchikey_scores[closely_related_inchikey14] for closely_related_inchikey14 in closest_library_structures)
            average_ms2deepscore_multiple_library_structures = sum_of_average_ms2ds_multiple_library_structures/ len(closest_library_structures)
            results_per_inchikey[inchikey] = (average_ms2deepscore_multiple_library_structures, average_tanimoto_score_multiple_library_spectra)
        return results_per_inchikey

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


def get_ms2query_model_prediction_single_spectrum(
        result_table: Union[ResultsTable, None],
        ms2query_nn_model
        ) -> ResultsTable:
    """Adds ms2query predictions to result table

    result_table:
    ms2query_model_file_name:
        File name of a hdf5 name containing the ms2query model.
    """
    # .values removes feature names, since this will cause a warning since the model was trained without feature names.
    current_query_matches_info = result_table.get_training_data().copy().values
    predictions = ms2query_nn_model.predict(current_query_matches_info)

    result_table.add_ms2query_meta_score(predictions)
    return result_table


def create_library_object_from_one_dir(directory_containing_library_and_models: str) -> MS2Library:
    """Selects file names corresponding to file types in a directory and creates an MS2Library object

    Args:
    ------
    directory:
        Path to the directory in which the files are stored
    """
    assert os.path.exists(directory_containing_library_and_models) \
           and not os.path.isfile(directory_containing_library_and_models), "Expected a directory"
    dict_with_file_names = \
        {"sqlite": None, "classifiers": None, "s2v_model": None, "ms2ds_model": None,
         "ms2query_model": None, "s2v_embeddings": None, "ms2ds_embeddings": None}

    # Go through spectra files in directory
    for file_name in os.listdir(directory_containing_library_and_models):
        file_path = os.path.join(directory_containing_library_and_models, file_name)
        # skip folders
        if os.path.isfile(file_path):
            if str.endswith(file_path, ".sqlite"):
                file_name = "sqlite"
                assert dict_with_file_names[file_name] is None, \
                    f"Multiple files could be the file containing the {file_name} file"
                dict_with_file_names[file_name] = file_path
            elif str.endswith(file_path, "ms2ds_embeddings.pickle"):
                file_name = "ms2ds_embeddings"
                assert dict_with_file_names[file_name] is None, \
                    f"Multiple files could be the file containing the {file_name} file"
                dict_with_file_names[file_name] = file_path
            elif str.endswith(file_path, "s2v_embeddings.pickle"):
                file_name = "s2v_embeddings"
                assert dict_with_file_names[file_name] is None, \
                    f"Multiple files could be the file containing the {file_name} file"
                dict_with_file_names[file_name] = file_path
            elif str.endswith(file_path, ".hdf5"):
                file_name = "ms2ds_model"
                assert dict_with_file_names[file_name] is None, \
                    f"Multiple files could be the file containing the {file_name} file"
                dict_with_file_names[file_name] = file_path
            elif str.endswith(file_path, ".pickle") and "ms2q" in file_name:
                file_name = "ms2query_model"
                assert dict_with_file_names[file_name] is None, \
                    f"Multiple files could be the file containing the {file_name} file"
                dict_with_file_names[file_name] = file_path
            elif str.endswith(file_path, ".model"):
                file_name = "s2v_model"
                assert dict_with_file_names[file_name] is None, \
                    f"Multiple files could be the file containing the {file_name} file"
                dict_with_file_names[file_name] = file_path
            elif str.endswith(file_path, "CF_NPC_classes.txt"):
                file_name = "classifiers"
                assert dict_with_file_names[file_name] is None, \
                    f"Multiple files could be the file containing the {file_name} file"
                dict_with_file_names[file_name] = file_path
    for file_type, stored_file_name in dict_with_file_names.items():
        if file_type != "classifiers":
            assert stored_file_name is not None, \
                f"The file type {file_type} was not found in the directory"
        elif stored_file_name is None:
            print(f"The file type {file_type} was not found in the directory ")

    return MS2Library(dict_with_file_names["sqlite"],
                      dict_with_file_names["s2v_model"],
                      dict_with_file_names["ms2ds_model"],
                      dict_with_file_names["s2v_embeddings"],
                      dict_with_file_names["ms2ds_embeddings"],
                      dict_with_file_names["ms2query_model"],
                      dict_with_file_names["classifiers"])
