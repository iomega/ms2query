import os
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from matchms.Spectrum import Spectrum
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model as load_ms2ds_model
from spec2vec.vector_operations import calc_vector
from tqdm import tqdm
from ms2query.create_sqlite_database import make_sqlfile_wrapper
from ms2query.spectrum_processing import (create_spectrum_documents,
                                          minimal_processing_multiple_spectra)
from ms2query.utils import load_pickled_file, convert_files_to_matchms_spectrum_objects


class LibraryFilesCreator:
    """Class to build a MS2Query library from input spectra and trained
    MS2DeepScore as well as Spec2Vec models.

    For example:

    .. code-block:: python

        from ms2query import LibraryFilesCreator

        # Initiate Creator
        library_creator = LibraryFilesCreator(
            spectrums_file,
            output_file_sqlite="... folder and file name base...",
            'tanimoto_scores.pickle',
            'ms2ds_model.hdf5',
            'spec2vec_model.model'
            progress_bars=True)
        #
        library_creator.create_all_library_files()

    """
    def __init__(self,
                 spectra_file_name: str,
                 output_base_filename: str,
                 tanimoto_scores_file_name: str = None,
                 s2v_model_file_name: str = None,
                 ms2ds_model_file_name: str = None,
                 **settings):
        """Creates files needed to run queries on a library

        Parameters
        ----------
        spectra_file_name:
            File name of a file containing mass spectra. Accepted file types are: "mzML", "json", "mgf", "msp",
        "mzxml", "usi" or "pickle"
        output_base_filename:
            The file name used as base for new files that are created.
            The following extensions are added to the output_base_filename
            For sqlite file: ".sqlite"
            For ms2ds_embeddings: "_ms2ds_embeddings.pickle"
            For s2v_embeddings: "_s2v_embeddings.pickle"
        tanimoto_scores_file_name:
            File name of a pickled file containing a dataframe with tanimoto
            scores. If self.calculate_new_tanimoto_scores = True, this will
            be the file name of a new file in which the tanimoto scores will
            be stored.
        s2v_model_file_name:
            file name of a s2v model
        ms2ds_model_file_name:
            File name of a ms2ds model

        **settings:
            The following optional parameters can be defined.
        spectrum_id_column_name:
            The name of the column or key under which the spectrum id is
            stored. Default = "spectrumid"
        progress_bars:
            If True, a progress bar of the different processes is shown.
            Default = True.
        """
        self.settings = self._set_settings(settings, output_base_filename)
        self._check_for_existing_files()

        # Load in the spectra
        self.list_of_spectra = \
            self._load_spectra_and_minimal_processing(
                spectra_file_name)

        # Load in tanimoto scores
        if tanimoto_scores_file_name is None:
            self.tanimoto_scores = None
        else:
            assert os.path.exists(tanimoto_scores_file_name), "Tanimoto scores file does not exists"
            self.tanimoto_scores = load_pickled_file(tanimoto_scores_file_name)
        # Load in spec2vec model
        if s2v_model_file_name is None:
            self.s2v_model = None
        else:
            assert os.path.exists(s2v_model_file_name), "Spec2Vec model file does not exists"
            self.s2v_model = Word2Vec.load(s2v_model_file_name)
        # load in ms2ds model
        if ms2ds_model_file_name is None:
            self.ms2ds_model = None
        else:
            assert os.path.exists(ms2ds_model_file_name), "MS2Deepscore model file does not exists"
            self.ms2ds_model = load_ms2ds_model(ms2ds_model_file_name)

    @staticmethod
    def _set_settings(new_settings: Dict[str, Union[str, bool]],
                      output_base_filename: str
                      ) -> Dict[str, Union[str, bool]]:
        """Changes default settings to new_settings and creates file names

        Args:
        ------
        new_settings:
            Dictionary with settings that should be changed. Only the
            keys given in default_settings can be used and the type has to be
            the same as the type of the values in default settings.
        output_base_filename:
            The file name used as base for new files that are created.
            The following extensions are added to the output_base_filename
            For sqlite file: ".sqlite"
            For ms2ds_embeddings: "_ms2ds_embeddings.pickle"
            For s2v_embeddings: "_s2v_embeddings.pickle"
        """
        default_settings = {"progress_bars": True,
                            "spectrum_id_column_name": "spectrumid"}

        for attribute in new_settings:
            assert attribute in default_settings, \
                f"Invalid argument in constructor:{attribute}"
            assert isinstance(new_settings[attribute],
                              type(default_settings[attribute])), \
                f"Different type is expected for argument: {attribute}"
            default_settings[attribute] = new_settings[attribute]

        # Set file names of new file
        default_settings["output_file_sqlite"] = \
            output_base_filename + ".sqlite"
        default_settings["ms2ds_embeddings_file_name"] = \
            output_base_filename + "_ms2ds_embeddings.pickle"
        default_settings["s2v_embeddings_file_name"] = \
            output_base_filename + "_s2v_embeddings.pickle"

        return default_settings

    def _check_for_existing_files(self):
        assert not os.path.exists(self.settings["output_file_sqlite"]), \
            f"The file {self.settings['output_file_sqlite']} already exists," \
            f" choose a different output_base_filename"
        assert not os.path.exists(self.settings[
                                      'ms2ds_embeddings_file_name']), \
            f"The file {self.settings['ms2ds_embeddings_file_name']} " \
            f"already exists, choose a different output_base_filename"
        assert not os.path.exists(self.settings[
            "s2v_embeddings_file_name"]), \
            f"The file {self.settings['s2v_embeddings_file_name']} " \
            f"already exists, choose a different output_base_filename"

    def _load_spectra_and_minimal_processing(self,
                                             pickled_spectra_file_name: str
                                             ) -> List[Spectrum]:
        """Loads spectra from pickled file and does minimal processing

        Args:
        ------
        pickled_spectra_file_name:
            The file name of a pickled file containing a list of spectra.
        """
        # Loads the spectra from a pickled file
        list_of_spectra = convert_files_to_matchms_spectrum_objects(pickled_spectra_file_name)

        # Does normalization and filtering of spectra
        list_of_spectra = \
            minimal_processing_multiple_spectra(
                list_of_spectra,
                progress_bar=self.settings["progress_bars"])
        return list_of_spectra

    def create_all_library_files(self):
        """Creates files with embeddings and a sqlite file with spectra data
        """
        self.create_sqlite_file()
        self.store_s2v_embeddings()
        self.store_ms2ds_embeddings()

    def create_sqlite_file(self):
        assert self.tanimoto_scores is not None, "No tanimoto scores were provided"
        make_sqlfile_wrapper(
            self.settings["output_file_sqlite"],
            self.tanimoto_scores,
            self.list_of_spectra,
            columns_dict={"precursor_mz": "REAL"},
            progress_bars=self.settings["progress_bars"],
        )

    def store_ms2ds_embeddings(self):
        """Creates a pickled file with embeddings scores for spectra

        A dataframe with as index randomly generated spectrum indexes and as columns the indexes
        of the vector is converted to pickle.
        """
        assert not os.path.exists(self.settings[
                                      'ms2ds_embeddings_file_name']), \
            "Given ms2ds_embeddings_file_name already exists"
        assert self.ms2ds_model is not None, "No MS2deepscore model was provided"
        ms2ds = MS2DeepScore(self.ms2ds_model,
                             progress_bar=self.settings["progress_bars"])

        # Compute spectral embeddings
        embeddings = ms2ds.calculate_vectors(self.list_of_spectra)
        spectrum_ids = np.arange(0, len(self.list_of_spectra))
        all_embeddings_df = pd.DataFrame(embeddings, index=spectrum_ids)
        all_embeddings_df.to_pickle(self.settings[
                                        'ms2ds_embeddings_file_name'])

    def store_s2v_embeddings(self):
        """Creates and stored a dataframe with embeddings as pickled file

        A dataframe with as index randomly generated spectrum indexes and as columns the indexes
        of the vector is converted to pickle.
        """
        assert not os.path.exists(self.settings[
            "s2v_embeddings_file_name"]), \
            "Given s2v_embeddings_file_name already exists"
        assert self.s2v_model is not None, "No spec2vec model was specified"
        # Convert Spectrum objects to SpectrumDocument
        spectrum_documents = create_spectrum_documents(
            self.list_of_spectra,
            progress_bar=self.settings["progress_bars"])
        embeddings_dict = {}
        for spectrum_id, spectrum_document in tqdm(enumerate(spectrum_documents),
                                      desc="Calculating embeddings",
                                      disable=not self.settings[
                                          "progress_bars"]):
            embedding = calc_vector(self.s2v_model,
                                    spectrum_document,
                                    allowed_missing_percentage=100)
            embeddings_dict[spectrum_id] = embedding

        # Convert to pandas Dataframe
        embeddings_dataframe = pd.DataFrame.from_dict(embeddings_dict,
                                                      orient="index")
        embeddings_dataframe.to_pickle(self.settings[
            "s2v_embeddings_file_name"])
