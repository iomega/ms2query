"""
This script is not needed for normally running MS2Query, it is only needed to generate a new library or to train
new models
"""

import os
from pathlib import Path
from typing import List, Union, Dict
import matchms.filtering as msfilters
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from matchms import Spectrum
from matchms.Spectrum import Spectrum
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model as load_ms2ds_model
from spec2vec.vector_operations import calc_vector
from tqdm import tqdm
from ms2query.clean_and_filter_spectra import create_spectrum_documents
from ms2query.create_new_library.add_classifire_classifications import select_compound_classes
from ms2query.create_new_library.create_sqlite_database import initialize_tables, fill_spectrum_data_table, \
    fill_inchikeys_table, add_dataframe_to_sqlite
from ms2query.utils import return_non_existing_file_name


class LibraryFilesCreator:
    """Class to build a MS2Query library from input spectra and trained
    MS2DeepScore as well as Spec2Vec models.

    For example:

    .. code-block:: python

        from ms2query.library_files_creator import LibraryFilesCreator
        from ms2query.utils import convert_files_to_matchms_spectrum_objects
        spectrum_file_location =
        library_spectra = convert_files_to_matchms_spectrum_objects(spectrum_file_location)
        # Fill in the missing values:
        library_creator = LibraryFilesCreator(library_spectra,
                                              output_directory=,
                                              ms2ds_model_file_name=,
                                              s2v_model_file_name=,)
        library_creator.clean_up_smiles_inchi_and_inchikeys(do_pubchem_lookup=True)
        library_creator.clean_peaks_and_normalise_intensities_spectra()
        library_creator.remove_not_fully_annotated_spectra()
        library_creator.remove_wrong_ion_mode("positive")
        library_creator.create_all_library_files()
    """
    def __init__(self,
                 library_spectra: List[Spectrum],
                 sqlite_file_name: Union[str, Path],
                 s2v_model_file_name: str = None,
                 ms2ds_model_file_name: str = None,
                 compound_classes: Union[bool, pd.DataFrame, None] = True
                 ):
        """Creates files needed to run queries on a library

        Parameters
        ----------
        library_spectra:
            A list containing matchms spectra objects for the library spectra.
            To load in library spectra use ms2query.utils load_matchms_spectrum_objects_from_file
        output_directory:
            The directory in which the created library files are stored. The used file names are
            For sqlite file: "ms2query_library.sqlite"
            For ms2ds_embeddings: "ms2ds_embeddings.pickle"
            For s2v_embeddings: "s2v_embeddings.pickle"
        s2v_model_file_name:
            file name of a s2v model
        ms2ds_model_file_name:
            File name of a ms2ds model
        """
        # pylint: disable=too-many-arguments
        if os.path.exists(sqlite_file_name):
            raise FileExistsError("The sqlite file already exists")
        else:
            self.sqlite_file_name = sqlite_file_name

        # Load in spec2vec model
        if os.path.exists(s2v_model_file_name):
            self.s2v_model = Word2Vec.load(s2v_model_file_name)
        else:
            raise FileNotFoundError("Spec2Vec model file does not exists")
        # load in ms2ds model
        if os.path.exists(ms2ds_model_file_name):
            self.ms2ds_model = load_ms2ds_model(ms2ds_model_file_name)
        else:
            raise FileNotFoundError("MS2Deepscore model file does not exists")
        # Initialise spectra
        self.list_of_spectra = library_spectra

        # Run default filters
        self.list_of_spectra = [msfilters.default_filters(s) for s in tqdm(self.list_of_spectra,
                                                                           desc="Applying default filters to spectra")]
        self.compound_classes = self.add_compound_classes(compound_classes)
        if self.compound_classes is not None:
            self.additional_inchikey_columns = list(compound_classes.columns)
        else:
            self.additional_inchikey_columns = []

        self.progress_bars = True
        self.additional_metadata_columns = {"precursor_mz": "REAL"}

    def add_compound_classes(self,
                             compound_classes: Union[pd.DataFrame, bool, None]):
        """Calculates compound classes if True, otherwise uses given compound_classes
        """
        if compound_classes is True:
            compound_classes = select_compound_classes(self.list_of_spectra)
        elif compound_classes is not None and isinstance(compound_classes, pd.DataFrame):
            if not compound_classes.index.name == "inchikey":
                raise ValueError("Expected a pandas dataframe with inchikey as index name")
        elif compound_classes is False or compound_classes is None:
            compound_classes = None
        else:
            raise ValueError("Expected a dataframe or True or None for compound classes")
        return compound_classes

    def create_sqlite_file(self):
        """Wrapper to create sqlite file containing spectrum information needed for MS2Query

        Args:
        -------
        sqlite_file_name:
            Name of sqlite_file that should be created, if it already exists the
            tables are added. If the tables in this sqlite file already exist, they
            will be overwritten.
        list_of_spectra:
            A list with spectrum objects
        columns_dict:
            Dictionary with as keys columns that need to be added in addition to
            the default columns and as values the datatype. The defaults columns
            are spectrum_id, peaks, intensities and metadata. The additional
            columns should be the same names that are in the metadata dictionary,
            since these values will be automatically added in the function
            add_list_of_spectra_to_sqlite.
            Default = None results in the default columns.
        progress_bars:
            If progress_bars is True progress bars will be shown for the different
            parts of the progress.
        """
        if os.path.exists(self.sqlite_file_name):
            raise FileExistsError("The sqlite file already exists")
        initialize_tables(self.sqlite_file_name,
                          additional_metadata_columns_dict=self.additional_metadata_columns,
                          additional_inchikey_columns=self.additional_inchikey_columns)
        fill_spectrum_data_table(self.sqlite_file_name, self.list_of_spectra, progress_bar=self.progress_bars)

        fill_inchikeys_table(self.sqlite_file_name, self.list_of_spectra,
                             compound_classes=self.compound_classes,
                             progress_bars=self.progress_bars)

        add_dataframe_to_sqlite(self.sqlite_file_name,
                                'MS2Deepscore_embeddings',
                                create_ms2ds_embeddings(self.ms2ds_model, self.list_of_spectra, self.progress_bars), )
        add_dataframe_to_sqlite(self.sqlite_file_name,
                                'Spec2Vec_embeddings',
                                create_s2v_embeddings(self.s2v_model, self.list_of_spectra, self.progress_bars))


def create_ms2ds_embeddings(ms2ds_model,
                            list_of_spectra,
                            progress_bar=True):
    """Creates the ms2deepscore embeddings for all spectra

    A dataframe with as index randomly generated spectrum indexes and as columns the indexes
    of the vector is converted to pickle.
    """
    assert ms2ds_model is not None, "No MS2deepscore model was provided"
    ms2ds = MS2DeepScore(ms2ds_model,
                         progress_bar=progress_bar)
    # Compute spectral embeddings
    embeddings = ms2ds.calculate_vectors(list_of_spectra)
    spectrum_ids = np.arange(0, len(list_of_spectra))
    all_embeddings_df = pd.DataFrame(embeddings, index=spectrum_ids)
    return all_embeddings_df


def create_s2v_embeddings(s2v_model,
                          list_of_spectra,
                          progress_bar=True):
    """Creates and stored a dataframe with embeddings as pickled file

    A dataframe with as index randomly generated spectrum indexes and as columns the indexes
    of the vector is converted to pickle.
    """
    assert s2v_model is not None, "No spec2vec model was specified"
    # Convert Spectrum objects to SpectrumDocument
    spectrum_documents = create_spectrum_documents(
        list_of_spectra,
        progress_bar=progress_bar)
    embeddings_dict = {}
    for spectrum_id, spectrum_document in tqdm(enumerate(spectrum_documents),
                                               desc="Calculating embeddings",
                                               disable=not progress_bar):
        embedding = calc_vector(s2v_model,
                                spectrum_document,
                                allowed_missing_percentage=100)
        embeddings_dict[spectrum_id] = embedding

    # Convert to pandas Dataframe
    embeddings_dataframe = pd.DataFrame.from_dict(embeddings_dict,
                                                  orient="index")
    return embeddings_dataframe
