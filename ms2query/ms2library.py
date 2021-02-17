from ms2query.query_from_sqlite_database import get_spectra_from_sqlite, \
    get_tanimoto_score_for_inchikeys
from typing import List, Dict, Any, Tuple
from matchms.Spectrum import Spectrum
import pandas as pd
import numpy as np
import sqlite3
import pickle
from ms2deepscore.models import load_model as load_ms2ds_model
from ms2deepscore.models import SiameseModel
from ms2deepscore import MS2DeepScore
from matchms.similarity import CosineGreedy, ModifiedCosine
from gensim.models import Word2Vec
from tqdm import tqdm
from spec2vec import SpectrumDocument, Spec2Vec
from spec2vec.vector_operations import calc_vector, cosine_similarity_matrix
from ms2query.s2v_functions import \
    post_process_s2v
from tensorflow.keras.models import load_model as load_nn_model


class Ms2Library:
    def __init__(self,
                 sqlite_file_location: str,
                 s2v_model_file_name: str,
                 ms2ds_model_file_name: str,
                 pickled_s2v_embeddings_file_name: str,
                 pickled_ms2ds_embeddings_file_name: str,
                 neural_network_file_name: str,
                 **settings):
        """

        Args:
        -------
        sqlite_file_location:
            The location at which the sqlite_file_is_stored. The file is
            expected to have 3 tables: tanimoto_scores, inchikeys and
            spectra_data. If no sqlite file is available paramater is expected
            to be None.
        file_name_dict:
            A dictionary containing the files needed to create a sql database.
            The dictionary is expected to contain the keys:
            "npy_file_location", "pickled_file_name" and "csv_file_name". With
            as value the location of these files.
            Default = None
        """
        # Set default settings
        self.mass_tolerance = 1.0

        # todo check if these scores can be taken from the nn model in some way
        #  since they should be the same as how the network is trained
        self.cosine_score_tolerance = 0.1
        self.base_nr_mass_similarity = 0.8
        # todo make new model that has a fixed basic mass
        self.max_parent_mass = 13418.370894192036

        # Change default settings to values given in **settings
        self._set_settings(settings)

        # todo check if the sqlite file contains the correct tables
        self.sqlite_file_location = sqlite_file_location
        self.s2v_model = Word2Vec.load(s2v_model_file_name)
        self.ms2ds_model = load_ms2ds_model(ms2ds_model_file_name)
        self.nn_model = load_nn_model(neural_network_file_name)
        # loads the library embeddings into memory
        with open(pickled_s2v_embeddings_file_name, "rb") as \
                pickled_s2v_embeddings:
            self.s2v_embeddings = pickle.load(pickled_s2v_embeddings)
        with open(pickled_ms2ds_embeddings_file_name, "rb") as \
                pickled_ms2ds_embeddings:
            self.ms2ds_embeddings = pickle.load(pickled_ms2ds_embeddings)

    def _set_settings(self,
                      settings: Dict[str, Any]):
        """Changes default settings to settings

        Attributes specified in settings are expected to have been defined with
        a default value before calling this function.
        Args
        ------
        settings:
            Dictionary with as keys the name of the attribute that should be
            set and as value the value this attribute should have.

        """
        # Get all attributes to check if the arguments in settings are allowed
        allowed_arguments = self.__dict__
        # Set all kwargs as attributes, when in allowed_arguments
        for key in settings:
            assert key in allowed_arguments, \
                f"Invalid argument in constructor:{key}"
            assert isinstance(settings[key], type(allowed_arguments[key])), \
                f"Different type is expected for argument: {key}"
            setattr(self, key, settings[key])

    def pre_select_spectra(self,
                           query_spectra: List[Spectrum]
                           ) -> Dict[str, List[str]]:
        """Returns dict with spectrum IDs that are preselected

        The keys are the query spectrum_ids and the values a list of the
        preselected spectrum_ids for each query spectrum.

        Args:
        ------
        query_spectra:
            spectra for which a preselection of possible library matches should
            be done
        """
        # todo decide how to do the preselection and use spec2vec and or
        #  ms2deepscore for this as well
        # spec2vec_similarities_scores = self._get_spec2vec_similarity_matrix(
        #     query_spectra)
        # ms2ds_similarities_scores = self._get_ms2deepscore_similarity_matrix(
        #     query_spectra)
        spectra_with_similar_masses = \
            self._get_parent_mass_matches_all_queries(query_spectra)

        dict_with_preselected_spectra = {}
        for spectrum_id in spectra_with_similar_masses:
            # select the spectra with similar masses as preselected matches
            # Later will be decided which spectra are actually sellected
            preselected_matches = spectra_with_similar_masses[spectrum_id]

            # todo remove this part once the preselection is done before
            #  loading to sqlite
            # Remove spectra that do not get through the post_process_selection
            spectra_to_skip = create_spectrum_documents(
                get_spectra_from_sqlite(self.sqlite_file_location,
                                        preselected_matches))[1]
            preselected_matches = [spectrum_id for spectrum_id
                                   in preselected_matches
                                   if spectrum_id not in spectra_to_skip]

            dict_with_preselected_spectra[spectrum_id] = preselected_matches
        return dict_with_preselected_spectra

    def _get_ms2deepscore_similarity_matrix(
            self,
            query_spectra) -> pd.DataFrame:
        """Returns a dataframe with the ms2deepscore similarity scores

        query_spectra:
            All query spectra that should get a similarity score with all
            library spectra.
        library_embeddings_file_name:
            File name of the pickled file in which the library embeddings are
            stored.
        """
        ms2ds = MS2DeepScore(self.ms2ds_model)
        query_embeddings = ms2ds.calculate_vectors(query_spectra)
        ms2ds_embeddings_numpy = self.ms2ds_embeddings.to_numpy()
        similarity_matrix = cosine_similarity_matrix(ms2ds_embeddings_numpy,
                                                     query_embeddings)
        similarity_matrix_dataframe = pd.DataFrame(
            similarity_matrix,
            index=self.ms2ds_embeddings.index,
            columns=[spectrum.get("spectrum_id") for
                     spectrum in query_spectra])
        return similarity_matrix_dataframe

    def _get_spec2vec_similarity_matrix(self,
                                        query_spectra: List[Spectrum]
                                        ) -> pd.DataFrame:
        """
        Returns s2v similarity scores between all query and library spectra

        The column names are the query spectrum ids and the indexes are the
        library spectrum ids.

        Args:
        ------
        query_spectra:
            All spectra for which similarity scores should be calculated.
        """

        # Convert list of Spectrum objects to list of SpectrumDocuments
        query_spectrum_documents = create_spectrum_documents(query_spectra)[0]

        embedding_dim_s2v = self.s2v_model.wv.vector_size
        query_embeddings = np.empty((len(query_spectrum_documents),
                                    embedding_dim_s2v))

        for i, spectrum_document in enumerate(query_spectrum_documents):
            # Get the embeddings for current spectra
            query_embeddings[i, :] = calc_vector(self.s2v_model,
                                                 spectrum_document)

        # Get the spec2vect cosine similarity score for all query spectra
        spec2vec_similarities = cosine_similarity_matrix(
            self.s2v_embeddings.to_numpy(),
            query_embeddings)
        # Convert to dataframe, with the correct indexes and columns.
        spec2vec_similarities_dataframe = pd.DataFrame(
            spec2vec_similarities,
            index=self.s2v_embeddings.index,
            columns=[s.get("spectrum_id") for s in query_spectrum_documents])
        return spec2vec_similarities_dataframe

    def _get_parent_mass_matches_all_queries(self,
                                             query_spectra: List[Spectrum]
                                             ) -> Dict[str, List[str]]:
        """Returns a dictionary with all library spectra with similar masses

        The keys are the query spectrum Ids and the values are the library
        spectrum Ids that have a similar parent mass.

        Args:
        -------
        query_spectra:
            Query spectra for which the library spectra with similar parent
            masses are returned.
        mass_tolerance: float, optional
            Specify tolerance for a parentmass match. Default = 1.
        """
        spectra_with_similar_mass_dict = {}
        conn = sqlite3.connect(self.sqlite_file_location)
        for query_spectrum in query_spectra:
            query_mass = query_spectrum.get("parent_mass")
            query_spectrum_id = query_spectrum.get("spectrum_id")

            # Get spectrum_ids from sqlite with parent_mass within tolerance
            sqlite_command = \
                f"""SELECT spectrum_id FROM spectrum_data
                    WHERE parent_mass > {query_mass - self.mass_tolerance}
                    AND parent_mass < {query_mass + self.mass_tolerance}"""

            cur = conn.cursor()
            cur.execute(sqlite_command)
            # Makes sure the output is a list of strings instead of
            # list of tuples
            cur.row_factory = lambda cursor, row: row[0]
            spectra_with_similar_mass = cur.fetchall()

            spectra_with_similar_mass_dict[query_spectrum_id] = \
                spectra_with_similar_mass
        conn.close()
        return spectra_with_similar_mass_dict

    def filter_by_tanimoto_prediction(self,
                                      matches_info: Dict[str, pd.DataFrame]
                                      ):
        """"""
        # todo Move to subclass together with self.nn_model
        for query_spectrum_id in matches_info:
            current_query_matches_info = matches_info[query_spectrum_id]
            prediction = self.nn_model.predict(current_query_matches_info)
            current_query_matches_info["tanimoto_prediction"] = prediction
            matches_info[query_spectrum_id] = current_query_matches_info
        # Add function that removes matches below a certain tanimoto prediction
        return matches_info

    def collect_matches_data_multiple_spectra(self,
                                              query_spectra: List[Spectrum]
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
        """
        # Gets a preselection of spectra for all query_spectra
        dict_with_preselected_spectra = self.pre_select_spectra(query_spectra)

        # Run neural network model over all found spectra and add the predicted
        # scores to a dict with the spectra in dataframes
        dict_with_preselected_spectra_info = {}
        for query_spectrum in query_spectra:
            spectrum_id = query_spectrum.get("spectrum_id")
            matches_with_info = \
                self.collect_data_for_tanimoto_prediction_model(
                    query_spectrum,
                    dict_with_preselected_spectra[spectrum_id])
            dict_with_preselected_spectra_info[spectrum_id] = matches_with_info
        return dict_with_preselected_spectra_info

    def collect_data_for_tanimoto_prediction_model(
            self,
            query_spectrum: Spectrum,
            preselected_spectrum_ids: List[str]) -> pd.DataFrame:
        """Returns dataframe with relevant info for nn model"""
        # Gets a list of all preselected spectra as Spectrum objects
        preselected_spectra_list = get_spectra_from_sqlite(
            self.sqlite_file_location,
            [spectrum_id for spectrum_id
             in preselected_spectrum_ids])

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

        # Get s2v_scores
        s2v_scores = Spec2Vec(self.s2v_model).matrix(
            create_spectrum_documents(preselected_spectra_list)[0],
            create_spectrum_documents([query_spectrum])[0])[:, 0]

        parent_masses = [spectrum.get("parent_mass")
                         for spectrum in preselected_spectra_list]
        normalized_parent_masses = [parent_mass/self.max_parent_mass
                                    for parent_mass in parent_masses]

        mass_similarity = [self.base_nr_mass_similarity **
                           (spectrum.get("parent_mass") -
                            query_spectrum.get("parent_mass"))
                           for spectrum in preselected_spectra_list]

        # Add info together into a dataframe
        preselected_spectra_df = pd.DataFrame(
            {"cosine_score": cosine_score,
             "cosine_matches": normalized_cosine_matches,
             "mod_cosine_score": mod_cosine_score,
             "mod_cosine_matches": normalized_mod_cos_matches,
             "s2v_scores": s2v_scores,
             "parent_mass": normalized_parent_masses,
             "mass_sim": mass_similarity},
            index=preselected_spectrum_ids)
        return preselected_spectra_df

    def get_tanimoto_scores(self,
                            list_of_inchikeys: List[str]
                            ) -> pd.DataFrame:
        """Returns a panda dataframe with the tanimoto scores"""
        tanimoto_score_matrix = get_tanimoto_score_for_inchikeys(
            list_of_inchikeys,
            self.sqlite_file_location)
        return tanimoto_score_matrix


# Not part of the class, used to create embeddings, that are than stored in a
# pickled file. (Storing in pickled file is not part of the function)

def store_ms2ds_embeddings(spectrum_list: List[Spectrum],
                           model: SiameseModel,
                           new_pickled_embeddings_file_name):
    """Creates a pickled file with embeddings scores for spectra

    A dataframe with as index the spectrum_ids and as columns the indexes of
    the vector is converted to pickle.

    Args:
    ------
    spectrum_list:
        Spectra for which embeddings should be calculated.
    model:
        SiameseModel that is used to calculate the embeddings.
    new_picled_embeddings_file_name:
        The file name in which the pickled dataframe is stored.
    """
    ms2ds = MS2DeepScore(model)
    spectra_vectors = ms2ds.calculate_vectors(spectrum_list)

    spectra_vector_dataframe = pd.DataFrame(
        spectra_vectors,
        index=[spectrum.get("spectrum_id") for spectrum in spectrum_list])
    spectra_vector_dataframe.to_pickle(new_pickled_embeddings_file_name)


def store_s2v_embeddings(spectra_list: List[Spectrum],
                         model: Word2Vec,
                         new_pickled_embeddings_file_name: str,
                         progress_bars: bool = True
                         ):
    """Creates a pickled file with embeddings for all given spectra

    A dataframe with as index the spectrum_ids and as columns the indexes of
    the vector is converted to pickle.

    Args
    ------
    spectra_list:
        Spectra for which the embeddings should be obtained
    model:
        Trained Spec2Vec model
    new_pickled_embeddings_file_name:
        File name for file created
    progress_bars:
        When True progress bars and steps in progress will be shown.
        Default = True
    """
    # Convert Spectrum objects to SpectrumDocument
    spectrum_documents = create_spectrum_documents(spectra_list,
                                                   progress_bar=progress_bars
                                                   )[0]
    embeddings_dict = {}
    for spectrum_document in tqdm(spectrum_documents,
                                  desc="Calculating embeddings",
                                  disable=not progress_bars):
        embedding = calc_vector(model,
                                spectrum_document)
        embeddings_dict[spectrum_document.get("spectrum_id")] = embedding

    # Convert to pandas Dataframe
    embeddings_dataframe = pd.DataFrame.from_dict(embeddings_dict,
                                                  orient="index")
    embeddings_dataframe.to_pickle(new_pickled_embeddings_file_name)


def create_spectrum_documents(query_spectra: List[Spectrum],
                              progress_bar: bool = False
                              ) -> Tuple[List[SpectrumDocument], List[str]]:
    """Transforms list of Spectrum to List of SpectrumDocument

    Args
    ------
    query_spectra:
        List of Spectrum objects that are transformed to SpectrumDocument
    progress_bar:
        When true a progress bar is shown. Default = False
    """
    # todo use new preprocessing and remove spectra_not_past_post_process_spect
    spectrum_documents = []
    spectra_not_past_post_process_spectra = []
    for spectrum in tqdm(query_spectra,
                         desc="Converting Spectrum to Spectrum_document",
                         disable=not progress_bar):
        post_process_spectrum = post_process_s2v(spectrum)
        if post_process_spectrum is not None:
            spectrum_documents.append(SpectrumDocument(
                post_process_spectrum,
                n_decimals=2))
        else:
            spectrum_id = spectrum.metadata["spectrum_id"]
            spectra_not_past_post_process_spectra.append(spectrum_id)
    return spectrum_documents, spectra_not_past_post_process_spectra


if __name__ == "__main__":
    # # To run a sqlite file should be made that contains also the table
    # # parent_mass. Use make_sqlfile_wrapper for this, see test_sqlite.py for
    # example (but use different start files for full dataset)
    sqlite_file_name = \
        "../downloads/data_all_inchikeys_with_tanimoto_and_parent_mass.sqlite"
    s2v_model_file_name = \
        "../downloads/" \
        "spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
    s2v_pickled_embeddings_file = \
        "../downloads/embeddings_all_spectra.pickle"
    ms2ds_model_file_name = \
        "../../ms2deepscore/data/" \
        "ms2ds_siamese_210207_ALL_GNPS_positive_L1L2.hdf5"
    ms2ds_embeddings_file_name = \
        "../../ms2deepscore/data/ms2ds_embeddings_2_spectra.pickle"
    neural_network_model_file_location = \
        "../model/nn_2000_queries_trimming_simple_10.hdf5"

    # Create library object
    my_library = Ms2Library(sqlite_file_name,
                            s2v_model_file_name,
                            ms2ds_model_file_name,
                            s2v_pickled_embeddings_file,
                            ms2ds_embeddings_file_name,
                            neural_network_model_file_location)
    # Get two query spectras
    query_spectra_to_test = get_spectra_from_sqlite(sqlite_file_name,
                                                    ["CCMSLIB00000001655"])

    print(my_library.collect_matches_data_multiple_spectra(
        query_spectra_to_test))

    # library_spectra = get_spectra_from_sqlite(sqlite_file_name,
    #                                           ["CCMSLIB00000001547",
    #                                            "CCMSLIB00000001549"],
    #                                           get_all_spectra=False,
    #                                           progress_bar=True)
    #
    # model = load_ms2ds_model(ms2ds_model_file_name)
    #
    # store_ms2ds_embeddings(library_spectra,
    #                        model,
    #                        ms2ds_embeddings_file_name)
