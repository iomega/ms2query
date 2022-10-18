import numpy as np
import sys
from typing import List, Tuple, Union
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from matchms import Spectrum, calculate_scores
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from ms2query import MS2Library, ResultsTable
from ms2query.query_from_sqlite_database import get_metadata_from_sqlite
from ms2query.spectrum_processing import minimal_processing_multiple_spectra
from ms2query.create_new_library.create_sqlite_database import add_fingerprint
from ms2query.create_new_library.library_files_creator import LibraryFilesCreator
from ms2query.utils import load_matchms_spectrum_objects_from_file
from ms2query.utils import load_pickled_file
from ms2query.create_new_library.train_ms2deepscore import calculate_tanimoto_scores

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


class DataCollectorForTraining():
    """Class to collect data needed to train a ms2query random forest"""
    def __init__(self,
                 ms2library: MS2Library,
                 preselection_cut_off: int = 2000):
        """Parameters
        ----------
        ms2library:
            A ms2library object. These should not contain the query spectra.
        preselection_cut_off:
            The number of highest scoring matches of MS2Deepscore that are used. For these top library matches all
            scores are calculated
        """
        # pylint: disable=too-many-arguments
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
        all_ms2ds_scores = self.ms2library._get_all_ms2ds_scores(query_spectra)
        for i, query_spectrum in tqdm(enumerate(query_spectra),
                                      desc="Get scores and tanimoto scores",
                                      disable=not self.ms2library.settings["progress_bars"]):
            results_table = ResultsTable(
                preselection_cut_off=self.preselection_cut_off,
                ms2deepscores=all_ms2ds_scores.iloc[:, i],
                query_spectrum=query_spectrum,
                sqlite_file_name=self.ms2library.sqlite_file_name)

            results_table = self.ms2library._calculate_features_for_random_forest_model(results_table)
            library_spectrum_ids = list(results_table.data.index)
            # Select the features (remove inchikey, since this should not be
            # used for training
            features_dataframe = results_table.get_training_data()
            # Get tanimoto scores
            tanimoto_scores = self.calculate_tanimoto_scores(query_spectrum, library_spectrum_ids)
            all_tanimoto_scores = \
                all_tanimoto_scores.append(tanimoto_scores,
                                           ignore_index=True)

            # Add matches for which a tanimoto score could be calculated
            matches_with_tanimoto = features_dataframe.loc[
                tanimoto_scores.index]
            info_of_matches_with_tanimoto = \
                info_of_matches_with_tanimoto.append(matches_with_tanimoto,
                                                     ignore_index=True)
        return info_of_matches_with_tanimoto, all_tanimoto_scores

    def calculate_tanimoto_scores(self, query_spectrum: Spectrum,
                                      spectra_ids_list: List[str]):
        # Get inchikeys belonging to spectra ids
        metadata_dict = get_metadata_from_sqlite(
            self.ms2library.sqlite_file_name,
            spectra_ids_list)
        query_spectrum_fingerprint = add_fingerprint(query_spectrum, fingerprint_type="daylight", nbits=2048).get("fingerprint")
        fingerprints = []
        for spectrum_id in spectra_ids_list:
            smiles = metadata_dict[spectrum_id]["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            fingerprint = np.array(Chem.RDKFingerprint(mol, fpSize=2048))
            assert isinstance(fingerprint, np.ndarray) and fingerprint.sum() > 0, \
                f"Fingerprint for 1 spectrum could not be set smiles is {smiles}"
            fingerprints.append(fingerprint)
        query_spectrum_fingerprint = np.array([query_spectrum_fingerprint])
        fingerprints = np.array(fingerprints)
        # Specify type and calculate similarities
        tanimoto_scores = jaccard_similarity_matrix(fingerprints, query_spectrum_fingerprint)
        tanimoto_df = pd.DataFrame(tanimoto_scores, index=spectra_ids_list, columns=["Tanimoto_score"])
        return tanimoto_df
