from ms2query.query_from_sqlite_database import get_spectra_from_sqlite, \
    get_tanimoto_score_for_inchikeys
from typing import List, Dict
from matchms.Spectrum import Spectrum
import pandas as pd
import numpy as np
import sqlite3
import pickle
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
                 pickled_embeddings_file_name: str,
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
        self._set_settings(settings)

        # todo check if the sqlite file contains the correct tables
        self.sqlite_file_location = sqlite_file_location
        self.s2v_model = Word2Vec.load(s2v_model_file_name)
        # loads the library embeddings into memory
        with open(pickled_embeddings_file_name, "rb") as pickled_embeddings:
            self.embeddings = pickle.load(pickled_embeddings)

    def _set_settings(self, settings):
        # Set default settings
        self.mass_tolerance = 1.0

        # Change default settings to settings given as kwargs
        allowed_arguments = self.__dict__
        for key in settings:
            assert key in allowed_arguments, \
                f"Invalid argument in constructor:{key}"
            assert isinstance(settings[key], allowed_arguments[key]), \
                f"Different type is expected for argument: {key}"
            setattr(self, key, settings[key])

    def pre_select_spectra(self,
                           query_spectra: List[Spectrum]):
        """Currently only runs functions that will later be needed to do the
        pre selection"""

        spec2vec_similarities_scores = self._get_spec2vec_similarity_matrix(
            query_spectra)
        same_masses = self._get_parent_mass_matches_all_queries(query_spectra)
        return spec2vec_similarities_scores, same_masses

    def _get_ms2deepscore_similarity_matrix(self):

        pass

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
            self.embeddings.to_numpy(),
            np.array(query_embeddings_list))
        # Convert to dataframe, with the correct indexes and columns.
        spec2vec_similarities_dataframe = pd.DataFrame(
            spec2vec_similarities,
            index=self.embeddings.index,
            columns=query_spectra_name_list)
        return spec2vec_similarities_dataframe

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


# Not part of the class, used to create embeddings, that are than stored in a
# pickled file. (Storing in pickled file is not part of the function)
def create_all_s2v_embeddings(sqlite_file_location: str,
                              model: Word2Vec,
                              progress_bars: bool = True
                              ) -> pd.DataFrame:
    """Returns a dataframe with embeddings for all library spectra

    Args
    ------
    sqlite_file_location:
        Location of sqlite file containing spectrum data.\
    model:
        Trained Spec2Vec model
    progress_bars:
        When True progress bars and steps in progress will be shown.
        Default = True
    """
    if progress_bars:
        print("Loading data from sqlite")
    library_spectra = get_spectra_from_sqlite(sqlite_file_location,
                                              [],
                                              get_all_spectra=True,
                                              progress_bar=progress_bars)
    # Convert Spectrum objects to SpectrumDocument
    spectrum_documents = create_spectrum_documents(library_spectra,
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
    return embeddings_dataframe


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
    # To run a sqlite file should be made that contains also the table
    # parent_mass. Use make_sqlfile_wrapper for this, see test_sqlite.py for
    # example (but use different start files for full dataset)
    sqlite_file_name = "../downloads/data_all_inchikeys_with_tanimoto_and_parent_mass.sqlite"
    model_file_name = "../downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
    new_pickled_embeddings_file = "../downloads/embeddings_all_spectra.pickle"
    model = Word2Vec.load(model_file_name)

    # Create pickled file with library embeddings:
    # library_embeddings = create_all_s2v_embeddings(
    #     sqlite_file_name,
    #     model)
    # print(library_embeddings)
    # library_embeddings.to_pickle(new_pickled_embeddings_file)

    # Create library object
    my_library = Ms2Library(sqlite_file_name, model_file_name,
                            new_pickled_embeddings_file)

    # Get two query spectras
    query_spectra_to_test = my_library.get_spectra(["CCMSLIB00000001547",
                                            "CCMSLIB00000001549"])

    s2v_matrix, similar_mass_dict = my_library.pre_select_spectra(
        query_spectra_to_test)

    print(s2v_matrix)
    print(similar_mass_dict)
