from ms2query.query_from_sqlite_database import get_spectra_from_sqlite, \
    get_tanimoto_score_for_inchikeys
from typing import List, Dict, Any
from matchms.Spectrum import Spectrum
import pandas as pd
import numpy as np
import sqlite3
import pickle
from ms2deepscore.models import load_model as load_ms2ds_model
from ms2deepscore.models import SiameseModel
from ms2deepscore import MS2DeepScore
from ms2deepscore.vector_operations import cosine_similarity_matrix
from gensim.models import Word2Vec
from tqdm import tqdm
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector, cosine_similarity_matrix
from ms2query.ms2query.s2v_functions import \
    post_process_s2v

# todo Check for _obj.get() for spectrumdocuments and after new release change
#  to .get()


class Ms2Library:
    def __init__(self,
                 sqlite_file_location: str,
                 s2v_model_file_name: str,
                 ms2ds_model_file_name: str,
                 pickled_s2v_embeddings_file_name: str,
                 pickled_ms2ds_embeddings_file_name: str,
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

        # Change default settings to values given in **settings
        self._set_settings(settings)

        # todo check if the sqlite file contains the correct tables
        self.sqlite_file_location = sqlite_file_location
        self.s2v_model = Word2Vec.load(s2v_model_file_name)
        self.ms2ds_model = load_ms2ds_model(ms2ds_model_file_name)
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
                           query_spectra: List[Spectrum]):
        """Returns dict of dataframe with preselected spectra

        The structure of the panda dataframe is

        Args:
        ------
        query_spectra:
            spectra for which a preselection of possible library matches should
            be done
        """

        # spec2vec_similarities_scores = self._get_spec2vec_similarity_matrix(
        #     query_spectra)
        same_masses = self._get_parent_mass_matches_all_queries(query_spectra)

        dict_with_preselected_spectra = {}
        for spectrum_id in same_masses:
            preselected_matches = same_masses[spectrum_id]
            dict_with_preselected_spectra[spectrum_id] = \
                pd.DataFrame(preselected_matches,
                             columns=["spectrum"])
        self.collect_data_for_tanimoto_prediction_model(dict_with_preselected_spectra)
        return dict_with_preselected_spectra

    def get_ms2deepscore_similarity_matrix(
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
        Returns a matrix with s2v similarity scores for all query spectra

        The column names are the query spectrum ids and the indexes are the
        library spectrum ids.

        Args:
        ------
        query_spectra:
            All spectra for which similarity scores should be calculated.
        """

        # Convert list of Spectrum objects to list of SpectrumDocuments
        query_spectrum_documents = create_spectrum_documents(query_spectra)

        query_spectra_name_list = []
        query_embeddings_list = []
        for spectrum_document in query_spectrum_documents:
            spectrum_name = spectrum_document._obj.get("spectrum_id")
            # Get the embeddings for current spectra
            query_spectrum_embedding = calc_vector(self.s2v_model,
                                                   spectrum_document)
            # Store in lists
            query_embeddings_list.append(query_spectrum_embedding)
            query_spectra_name_list.append(spectrum_name)

        # Get the spec2vect cosine similarity score for all query spectra
        spec2vec_similarities = cosine_similarity_matrix(
            self.s2v_embeddings.to_numpy(),
            np.array(query_embeddings_list))
        # Convert to dataframe, with the correct indexes and columns.
        spec2vec_similarities_dataframe = pd.DataFrame(
            spec2vec_similarities,
            index=self.s2v_embeddings.index,
            columns=query_spectra_name_list)
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

    def get_spectra(self, spectrum_id_list: List[str]) -> List[Spectrum]:
        """Returns the spectra corresponding to the spectrum ids"""
        spectra_list = get_spectra_from_sqlite(self.sqlite_file_location,
                                               spectrum_id_list)
        return spectra_list

    def get_tanimoto_scores(self,
                            list_of_inchikeys: List[str]
                            ) -> pd.DataFrame:
        """Returns a panda dataframe with the tanimoto scores"""
        tanimoto_score_matrix = get_tanimoto_score_for_inchikeys(
            list_of_inchikeys,
            self.sqlite_file_location)
        return tanimoto_score_matrix

    def collect_data_for_tanimoto_prediction_model(self,
                                                   preselected_spectra:
                                                   Dict[str, pd.DataFrame]):
        column_list = ["cosine_score",
                       "cosine_matches",
                       "mod_cosine_score",
                       "mod_cosine_matches",
                       "s2v_scores"]
        for query_spectrum_id in preselected_spectra:
            preselected_spectra_dataframe = preselected_spectra[query_spectrum_id]
            preselected_spectra_list = [spectrum_id for spectrum_id in preselected_spectra_dataframe['spectrum']]
            print(preselected_spectra_list)

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
                                                   progress_bar=progress_bars)
    embeddings_dict = {}
    for spectrum_document in tqdm(spectrum_documents,
                                  desc="Calculating embeddings",
                                  disable=not progress_bars):
        embedding = calc_vector(model,
                                spectrum_document)
        embeddings_dict[spectrum_document._obj.get("spectrum_id")] = embedding

    # Convert to pandas Dataframe
    embeddings_dataframe = pd.DataFrame.from_dict(embeddings_dict,
                                                  orient="index")
    embeddings_dataframe.to_pickle(new_pickled_embeddings_file_name)


def create_spectrum_documents(query_spectra: List[Spectrum],
                              progress_bar: bool = False
                              ) -> List[SpectrumDocument]:
    """Transforms list of Spectrum to dict of SpectrumDocument

    Keys are the spectrum_id and values the SpectrumDocument

    Args
    ------
    query_spectra:
        List of Spectrum objects that are transformed to SpectrumDocument
    progress_bar:
        When true a progress bar is shown. Default = False
    """
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
    return spectrum_documents


if __name__ == "__main__":
    # # To run a sqlite file should be made that contains also the table
    # # parent_mass. Use make_sqlfile_wrapper for this, see test_sqlite.py for
    # example (but use different start files for full dataset)
    sqlite_file_name = "../downloads/data_all_inchikeys_with_tanimoto_and_parent_mass.sqlite"
    s2v_model_file_name = "../downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
    s2v_pickled_embeddings_file = "../downloads/embeddings_all_spectra.pickle"
    ms2ds_model_file_name = "../../ms2deepscore/data/ms2ds_siamese_210207_ALL_GNPS_positive_L1L2.hdf5"
    ms2ds_embeddings_file_name = "../../ms2deepscore/data/ms2ds_embeddings_2_spectra.pickle"

    # Create library object
    my_library = Ms2Library(sqlite_file_name,
                            s2v_model_file_name,
                            ms2ds_model_file_name,
                            s2v_pickled_embeddings_file,
                            ms2ds_embeddings_file_name)
    # Get two query spectras
    query_spectra_to_test = my_library.get_spectra(["CCMSLIB00000001547",
                                                    "CCMSLIB00000001549"])

    my_library.pre_select_spectra(query_spectra_to_test)

    # library_spectra = get_spectra_from_sqlite(sqlite_file_name,
    #                                           ["CCMSLIB00000001547",
    #                                            "CCMSLIB00000001549"],
    #                                           get_all_spectra=False,
    #                                           progress_bar=True)
    #
    # model = load_ms2ds_model(ms2ds_model_file_name)
    #
    # store_ms2ds_embeddings(library_spectra, model, ms2ds_embeddings_file_name)
