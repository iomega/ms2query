import sys
from typing import List, Tuple, Union
import pandas as pd
from matchms.typing import SpectrumType
from tqdm import tqdm
from ms2query import MS2Library, ResultsTable
from ms2query.query_from_sqlite_database import get_metadata_from_sqlite
from ms2query.spectrum_processing import minimal_processing_multiple_spectra
from ms2query.utils import load_pickled_file


if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


class DataCollectorForTraining(MS2Library):
    """Class to collect data needed to train a ms2query random forest"""
    def __init__(self,
                 sqlite_file_location: str,
                 s2v_model_file_name: str,
                 ms2ds_model_file_name: str,
                 pickled_s2v_embeddings_file_name: str,
                 pickled_ms2ds_embeddings_file_name: str,
                 training_spectra_file: str,
                 validation_spectra_file: str,
                 tanimoto_scores_df_file_name: str,
                 preselection_cut_off: int = 2000,
                 **settings):
        """Parameters
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
        training_spectra_file:
            Pickled file with training spectra.
        validation_spectra_file:
            Pickled file with validation spectra.
        tanimoto_scores_df_file_name:
            A pickled file containing a dataframe with the tanimoto scores
            between all inchikeys. The tanimoto scores in SQLite cannot be
            used, since these do not contain the inchikeys for the training
            spectra.


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
        max_precursor_mz:
            The value used to normalize the precursor m/z by dividing it by the
            max_precursor_mz. Default = 13428.370894192036
        progress_bars:
            If True progress bars will be shown. Default = True"""
        # pylint: disable=too-many-arguments
        super().__init__(sqlite_file_location, s2v_model_file_name, ms2ds_model_file_name,
                         pickled_s2v_embeddings_file_name, pickled_ms2ds_embeddings_file_name, None, **settings)
        self.tanimoto_scores: pd.DataFrame = \
            load_pickled_file(tanimoto_scores_df_file_name)
        self.training_spectra = minimal_processing_multiple_spectra(
            load_pickled_file(training_spectra_file))
        self.validation_spectra = minimal_processing_multiple_spectra(
            load_pickled_file(validation_spectra_file))
        self.preselection_cut_off = preselection_cut_off

    def create_train_and_val_data(self,
                                  save_file_name: Union[None, str] = None
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame,
                                             pd.DataFrame, pd.DataFrame]:
        """Creates the training and validation sets and labels

        The sets contain the top 20 ms2ds matches of each spectrum and a
        collection of different scores and data of these matches in a
        pd.DataFrame. The labels contain a dataframe with the tanimoto scores.
        Args
        ----
        save_file_name:
            File name to which the result will be stored. The result is stored
            as a pickled file of a tuple containing the training_set, the
            training_labels, the validation_set and the validation_labels in
            that order.
            """
        training_set, training_labels = \
            self.get_matches_info_and_tanimoto(self.training_spectra)
        validation_set, validation_labels = \
            self.get_matches_info_and_tanimoto(self.validation_spectra)

        if save_file_name:
            with open(save_file_name, "wb") \
                    as new_file:
                pickle.dump((training_set,
                             training_labels,
                             validation_set,
                             validation_labels), new_file)
        return training_set, training_labels, validation_set, validation_labels

    def get_matches_info_and_tanimoto(self,
                                      query_spectra: List[SpectrumType]):
        """Returns tanimoto scores and info about matches of all query spectra

        A selection of matches is made for each query_spectrum. Based on the
        spectra multiple scores are calculated (info_of_matches_with_tanimoto)
        and the tanimoto scores based on the smiles is returned. All matches of
        all query_spectra are added together and the order of the tanimoto
        scores corresponds to the order of the info, so they can be used for
        training.

        Args:
        ------
        query_spectra:
            List of Spectrum objects
        """
        all_tanimoto_scores = pd.DataFrame()
        info_of_matches_with_tanimoto = pd.DataFrame()
        all_ms2ds_scores = self._get_all_ms2ds_scores(query_spectra)
        for i, query_spectrum in tqdm(enumerate(query_spectra),
                                      desc="Get scores and tanimoto scores",
                                      disable=not self.settings["progress_bars"]):
            results_table = ResultsTable(
                preselection_cut_off=self.preselection_cut_off,
                ms2deepscores=all_ms2ds_scores.iloc[:, i],
                query_spectrum=query_spectrum,
                sqlite_file_name=self.sqlite_file_name)

            results_table = self._calculate_features_for_random_forest_model(results_table)
            library_spectrum_ids = list(results_table.data.index)
            # Select the features (remove inchikey, since this should not be
            # used for training
            features_dataframe = results_table.get_training_data()
            # Get tanimoto scores, spectra that do not have an inchikey are not
            # returned.
            tanimoto_scores_for_query_spectrum = \
                self.get_tanimoto_for_spectrum_ids(query_spectrum,
                                                   library_spectrum_ids)
            all_tanimoto_scores = \
                all_tanimoto_scores.append(tanimoto_scores_for_query_spectrum,
                                           ignore_index=True)

            # Add matches for which a tanimoto score could be calculated
            matches_with_tanimoto = features_dataframe.loc[
                tanimoto_scores_for_query_spectrum.index]
            info_of_matches_with_tanimoto = \
                info_of_matches_with_tanimoto.append(matches_with_tanimoto,
                                                     ignore_index=True)
        # Converted to float32 since keras model cannot read float64
        return info_of_matches_with_tanimoto, all_tanimoto_scores

    def get_tanimoto_for_spectrum_ids(self,
                                      query_spectrum: SpectrumType,
                                      spectra_ids_list: List[str]
                                      ) -> pd.DataFrame:
        """Returns a dataframe with tanimoto scores

        Spectra in spectra_ids_list without inchikey are removed.
        Args:
        ------
        query_spectrum:
            Single Spectrum, the tanimoto scores are calculated between this
            spectrum and the spectra in match_spectrum_ids.
        match_spectrum_ids:
            list of spectrum_ids, which are preselected matches of the
            query_spectrum
        """
        query_inchikey14 = query_spectrum.get("inchikey")[:14]
        assert len(query_inchikey14) == 14, \
            f"Expected inchikey of length 14, " \
            f"got inchikey = {query_inchikey14}"

        # Get inchikeys belonging to spectra ids
        metadata_dict = get_metadata_from_sqlite(
            self.sqlite_file_name,
            spectra_ids_list,
            self.settings["spectrum_id_column_name"])
        unfiltered_inchikeys = [metadata_dict[spectrum_id]["inchikey"]
                                for spectrum_id in spectra_ids_list]

        inchikey14s_dict = {}
        for i, inchikey in enumerate(unfiltered_inchikeys):
            # Only get the first 14 characters of the inchikeys
            inchikey14 = inchikey[:14]
            spectrum_id = spectra_ids_list[i]
            # Don't save spectra that do not have an inchikey. If a spectra has
            # no inchikey it is stored as "", so it will not be stored.
            if len(inchikey14) == 14:
                inchikey14s_dict[spectrum_id] = inchikey14

        tanimoto_scores_spectra_ids = pd.DataFrame(
            columns=["Tanimoto_score"],
            index=list(inchikey14s_dict.keys()))
        for spectrum_id, inchikey14 in inchikey14s_dict.items():
            tanimoto_score = self.tanimoto_scores.loc[inchikey14,
                                                      query_inchikey14]
            tanimoto_scores_spectra_ids.at[spectrum_id, "Tanimoto_score"] = \
                tanimoto_score
        return tanimoto_scores_spectra_ids
