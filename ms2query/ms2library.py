from ms2query.query_from_sqlite_database import get_spectra_from_sqlite, \
    get_tanimoto_score_for_inchikeys
from typing import List, Dict, Union
from matchms.Spectrum import Spectrum
import pandas as pd
import numpy as np
import sqlite3
import ast
import pickle
from gensim.models import Word2Vec
from tqdm import tqdm
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector, cosine_similarity_matrix
from ms2query.ms2query.query_from_sqlite_database import convert_array
from ms2query.ms2query.s2v_functions import \
    post_process_s2v


class Ms2Library:
    def __init__(self,
                 sqlite_file_location: str,
                 model_file_name: str,
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
        # Set default settings
        self.mass_tolerance = 1.0

        # Change default settings to settings given as kwargs
        allowed_arguments = self.__dict__
        for key in settings:
            if key in allowed_arguments:
                if isinstance(settings[key], allowed_arguments[key]):
                    setattr(self, key, settings[key])
                else:
                    print(f"Different type is expected for argument: {key}")
            else:
                print(f"Invalid argument in constructor:{key}")

        # todo check if the sqlite file contains the correct tables
        self.sqlite_file_location = sqlite_file_location
        self.model = Word2Vec.load(model_file_name)
        # loads the library embeddings into memory
        with open(pickled_embeddings_file_name, "rb") as pickled_embeddings:
            self.embeddings = pickle.load(pickled_embeddings)

    def pre_select_spectra(self,
                           query_spectra: List[Spectrum]):
        def get_spec2vec_similarity_matrix() -> pd.DataFrame:
            """
            Returns a matrix with s2v similarity scores for all query spectra

            The column names are the query spectrum ids and the indexes are the
            library spectrum ids.

            Args:
            ------
            query_spectra:
                All spectra for which similarity scores should be calculated.
            """

            def create_spectrum_documents() -> Dict[str, SpectrumDocument]:
                """Transforms list of Spectrum to dict of SpectrumDocument

                Keys are the spectrum_id and values the SpectrumDocument"""
                spectrum_documents = {}
                for spectrum in query_spectra:
                    post_process_spectrum = post_process_s2v(spectrum)
                    spectrum_id = spectrum.metadata["spectrum_id"]
                    if post_process_spectrum is not None:
                        spectrum_documents[spectrum_id] = SpectrumDocument(
                            post_process_spectrum,
                            n_decimals=2)
                    else:
                        print(f"Spectrum {spectrum_id} did not pass "
                              f"post_process_spectrum")
                return spectrum_documents

            # Convert list of Spectrum objects to dict with SpectrumDocuments
            query_spectrum_documents = create_spectrum_documents()

            query_spectra_name_list = []
            query_embeddings_list = []
            for spectrum_name in query_spectrum_documents:
                spectrum_document = query_spectrum_documents[
                    spectrum_name]
                # Get the embeddings for current spectra
                query_spectrum_embedding = calc_vector(self.model,
                                                       spectrum_document)

                query_embeddings_list.append(query_spectrum_embedding)
                query_spectra_name_list.append(spectrum_name)

            query_embedding_ndarray = np.array(query_embeddings_list)
            library_embeddings_ndarray = self.embeddings.to_numpy()

            # Get the spec2vect cosine similarity score for all query spectra
            spec2vec_similarities = cosine_similarity_matrix(
                library_embeddings_ndarray,
                query_embedding_ndarray)
            # Convert to dataframe, with the correct indexes and columns.
            spec2vec_similarities_dataframe = pd.DataFrame(
                spec2vec_similarities,
                index=self.embeddings.index,
                columns=query_spectra_name_list)
            return spec2vec_similarities_dataframe

        def get_parent_mass_matches_all_queries() -> Dict[str, List[str]]:
            """Returns a dictionary with all spectra with similar masses

            The keys are the query spectrum Ids and the values are the library
            spectrum Ids that have a similar parent mass.

            Args:
            -------
            mass_tolerance: float, optional
                Specify tolerance for a parentmass match. Default = 1.
            """
            def get_parent_mass_matches() -> List[str]:
                """Returns all spectra ids that have a similar mass to query"""
                conn = sqlite3.connect(self.sqlite_file_location)
                # Get all relevant data.
                sqlite_command = \
                    f"""SELECT spectrum_id FROM spectrum_data
                        WHERE parent_mass > {query_mass - self.mass_tolerance}
                        AND parent_mass < {query_mass + self.mass_tolerance}"""

                cur = conn.cursor()
                cur.execute(sqlite_command)
                spectra_with_similar_mass = []
                for result in cur:
                    spectra_with_similar_mass.append(result[0])
                conn.close()
                return spectra_with_similar_mass

            spectra_with_similar_mass_dict = {}
            for query_spectrum in query_spectra:
                query_mass = query_spectrum.get("parent_mass")
                query_spectrum_id = query_spectrum.get("spectrum_id")
                spectra_with_similar_mass_dict[query_spectrum_id] = \
                    get_parent_mass_matches()
            return spectra_with_similar_mass_dict

        def get_ms2deepscore_similarity_matrix():
            pass

        spec2vec_similarities_scores = get_spec2vec_similarity_matrix()
        same_masses = get_parent_mass_matches_all_queries()
        return spec2vec_similarities_scores, same_masses

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
                              progress_bar: bool = True
                              ) -> pd.DataFrame:
    """Returns a dataframe with embeddings for all library spectra"""
    # todo replace sqlite lookup with
    #  get_spectra_from_sqlite(get_all_spectra = True)
    sqlite3.register_converter("array", convert_array)

    conn = sqlite3.connect(sqlite_file_location,
                           detect_types=sqlite3.PARSE_DECLTYPES)

    # Get all relevant data.
    sqlite_command = f"""SELECT peaks, intensities, metadata 
                         FROM spectrum_data"""

    cur = conn.cursor()
    cur.execute(sqlite_command)

    # Convert to list of matchms.Spectrum
    embeddings_dict = {}
    for result in tqdm(cur,
                       desc="Calculating embeddings",
                       disable=not progress_bar):
        peaks = result[0]
        intensities = result[1]
        metadata = ast.literal_eval(result[2])

        spectrum = Spectrum(mz=peaks,
                            intensities=intensities,
                            metadata=metadata)
        embedding = create_s2v_embedding(spectrum,
                                         model)
        if embedding is not None:
            embeddings_dict[metadata["spectrum_id"]] = embedding
    conn.close()
    embeddings_dataframe = pd.DataFrame.from_dict(embeddings_dict,
                                                  orient="index")
    return embeddings_dataframe


def create_spectrum_document(spectrum: Spectrum
                             ) -> Union[SpectrumDocument, None]:
    """Transforms a Spectrum object to a SpectrumDocument"""
    spectrum = post_process_s2v(spectrum)
    if spectrum is None:
        return None
    spectrum_document = SpectrumDocument(spectrum, n_decimals=2)
    return spectrum_document


def create_s2v_embedding(spectrum: Spectrum,
                         model
                         ) -> Union[np.array, None]:
    """Returns s2v embeddings for one spectrum"""
    spectrum_document = create_spectrum_document(spectrum)
    if spectrum_document is None:
        return None
    spectrum_embedding = calc_vector(model,
                                     spectrum_document)
    return spectrum_embedding


if __name__ == "__main__":
    # To run a sqlite file should be made that contains also the table
    # parent_mass. Use make_sqlfile_wrapper for this, see test_sqlite.py for
    # example (but use different start files for full dataset)
    sqlite_file_name = "../downloads/data_all_inchikeys_with_tanimoto_and_parent_mass.sqlite"
    model_file_name = "../downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
    new_pickled_embeddings_file = "../downloads/embeddings_all_spectra.pickle"
    model = Word2Vec.load(model_file_name)

    # Create pickled file with library embeddings:
    library_embeddings = create_all_s2v_embeddings(
        sqlite_file_name,
        model)

    library_embeddings.to_pickle(new_pickled_embeddings_file)

    # Create library object
    my_library = Ms2Library(sqlite_file_name,
                            model_file_name,
                            new_pickled_embeddings_file)

    # Get two query spectras
    query_spectra = my_library.get_spectra(["CCMSLIB00000001547",
                                            "CCMSLIB00000001549"])

    s2v_matrix, similar_mass_dict = my_library.pre_select_spectra(
        query_spectra)

    print(s2v_matrix)
    print(similar_mass_dict)
