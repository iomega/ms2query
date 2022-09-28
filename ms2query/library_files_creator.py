import os
from typing import List
import matchms.filtering as msfilters
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from matchms.Spectrum import Spectrum
from matchmsextras.pubchem_lookup import pubchem_metadata_lookup
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model as load_ms2ds_model
from spec2vec.vector_operations import calc_vector
from tqdm import tqdm
from ms2query.create_sqlite_database import make_sqlfile_wrapper
from ms2query.spectrum_processing import create_spectrum_documents


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
                                              ion_mode="positive",
                                              ms2ds_model_file_name=,
                                              s2v_model_file_name=,)
        library_creator.clean_up_smiles_inchi_and_inchikeys(do_pubchem_lookup=True)
        library_creator.clean_peaks_and_normalise_intensities_spectra()
        library_creator.remove_not_fully_annotated_spectra()
        library_creator.calculate_tanimoto_scores()
        library_creator.create_all_library_files()

    """
    def __init__(self,
                 library_spectra: List[Spectrum],
                 output_directory: str,
                 ion_mode: str = "positive",
                 s2v_model_file_name: str = None,
                 ms2ds_model_file_name: str = None,
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
        ion_mode:
            The ion mode of the library you want to create. Spectra not in this ion mode will be removed
        s2v_model_file_name:
            file name of a s2v model
        ms2ds_model_file_name:
            File name of a ms2ds model
        """
        # pylint: disable=too-many-arguments
        self.progress_bars = True
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
        self.sqlite_file_name = os.path.join(output_directory, "ms2query_library.sqlite")
        self.ms2ds_embeddings_file_name = os.path.join(output_directory, "ms2ds_embeddings.pickle")
        self.s2v_embeddings_file_name = os.path.join(output_directory, "s2v_embeddings.pickle")
        # These checks are performed at the start, since the filtering of spectra can take long
        self._check_for_existing_files()
        assert ion_mode in {"positive", "negative"}, "ion_mode should be set to 'positive' or 'negative'"
        self.ion_mode = ion_mode
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
        # Initialise spectra
        self.list_of_spectra = library_spectra

        # Run default filters
        self.list_of_spectra = [msfilters.default_filters(s) for s in tqdm(self.list_of_spectra,
                                                                           desc="Applying default filters to spectra")]
        # Remove spectra with the wrong ion mode (or no ion mode)
        self.remove_wrong_ion_modes()

    def _check_for_existing_files(self):
        assert not os.path.exists(self.sqlite_file_name), \
            f"The file {self.sqlite_file_name} already exists," \
            f" choose a different output_base_filename"
        assert not os.path.exists(self.ms2ds_embeddings_file_name), \
            f"The file {self.ms2ds_embeddings_file_name} " \
            f"already exists, choose a different output_base_filename"
        assert not os.path.exists(self.s2v_embeddings_file_name), \
            f"The file {self.s2v_embeddings_file_name} " \
            f"already exists, choose a different output_base_filename"

    def clean_up_smiles_inchi_and_inchikeys(self, do_pubchem_lookup):
        """Uses filters to clean ms2query

        do_pubchem_lookup: If true missing information will be searched on pubchem"""
        def run_metadata_filters(s):
            # Default filters
            s = msfilters.derive_adduct_from_name(s)
            s = msfilters.add_parent_mass(s, estimate_from_adduct=True)

            # Here, undefiend entries will be harmonized (instead of having a huge variation of None,"", "N/A" etc.)
            s = msfilters.harmonize_undefined_inchikey(s)
            s = msfilters.harmonize_undefined_inchi(s)
            s = msfilters.harmonize_undefined_smiles(s)

            # The repair_inchi_inchikey_smiles function will correct misplaced metadata
            # (e.g. inchikeys entered as inchi etc.) and harmonize the entry strings.
            s = msfilters.repair_inchi_inchikey_smiles(s)

            # Where possible (and necessary, i.e. missing): Convert between smiles, inchi, inchikey to complete metadata.
            # This is done using functions from rdkit.
            s = msfilters.derive_inchi_from_smiles(s)
            s = msfilters.derive_smiles_from_inchi(s)
            s = msfilters.derive_inchikey_from_inchi(s)

            if do_pubchem_lookup:
                s = pubchem_metadata_lookup(s,
                                            mass_tolerance=2.0,
                                            allowed_differences=[(18.03, 0.01),
                                                                 (18.01, 0.01)],
                                            name_search_depth=15)
            return s
        self.list_of_spectra = [run_metadata_filters(s) for s in tqdm(self.list_of_spectra,
                                                                      desc="Cleaning metadata library spectra")]

    def remove_wrong_ion_modes(self):
        spectra_to_keep = []
        for spec in tqdm(self.list_of_spectra, desc=f"Selecting {self.ion_mode} mode spectra"):
            if spec.get("ionmode") == self.ion_mode:
                spectra_to_keep.append(spec)
        print(f"From {len(self.list_of_spectra)} spectra, "
              f"{len(self.list_of_spectra) - len(spectra_to_keep)} are removed since they are not in {self.ion_mode} mode")
        self.list_of_spectra = spectra_to_keep

    def remove_not_fully_annotated_spectra(self):
        fully_annotated_spectra = []
        for spectrum in self.list_of_spectra:
            inchikey = spectrum.get("inchikey")
            if inchikey is not None and len(inchikey) > 13:
                smiles = spectrum.get("smiles")
                inchi = spectrum.get("inchi")
                if smiles is not None and len(smiles) > 0:
                    if inchi is not None and len(inchi) > 0:
                        fully_annotated_spectra.append(spectrum)
        print(f"From {len(self.list_of_spectra)} spectra, "
              f"{len(self.list_of_spectra) - len(fully_annotated_spectra)} are removed since they are not fully annotated")
        self.list_of_spectra = fully_annotated_spectra

    def clean_peaks_and_normalise_intensities_spectra(self):
        """Cleans library spectra

        pubchem_lookup:

        """
        def normalize_and_filter_peaks(spectrum):
            spectrum = msfilters.normalize_intensities(spectrum)
            spectrum = msfilters.select_by_relative_intensity(spectrum, 0.001, 1)
            spectrum = msfilters.select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)
            spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=500)
            spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=3)
            return spectrum

        library_spectra = [normalize_and_filter_peaks(s) for s in tqdm(self.list_of_spectra,
                                                                       desc="Cleaning and filtering peaks library spectra")]
        library_spectra = [s for s in library_spectra if s is not None]
        self.list_of_spectra = library_spectra

    def create_all_library_files(self):
        """Creates files with embeddings and a sqlite file with spectra data
        """
        self.create_sqlite_file()
        self.store_s2v_embeddings()
        self.store_ms2ds_embeddings()

    def create_sqlite_file(self):
        make_sqlfile_wrapper(
            self.sqlite_file_name,
            self.list_of_spectra,
            columns_dict={"precursor_mz": "REAL"},
            progress_bars=self.progress_bars,
        )

    def store_ms2ds_embeddings(self):
        """Creates a pickled file with embeddings scores for spectra

        A dataframe with as index randomly generated spectrum indexes and as columns the indexes
        of the vector is converted to pickle.
        """
        assert not os.path.exists(self.ms2ds_embeddings_file_name), \
            "Given ms2ds_embeddings_file_name already exists"
        assert self.ms2ds_model is not None, "No MS2deepscore model was provided"
        ms2ds = MS2DeepScore(self.ms2ds_model,
                             progress_bar=self.progress_bars)

        # Compute spectral embeddings
        embeddings = ms2ds.calculate_vectors(self.list_of_spectra)
        spectrum_ids = np.arange(0, len(self.list_of_spectra))
        all_embeddings_df = pd.DataFrame(embeddings, index=spectrum_ids)
        all_embeddings_df.to_pickle(self.ms2ds_embeddings_file_name)

    def store_s2v_embeddings(self):
        """Creates and stored a dataframe with embeddings as pickled file

        A dataframe with as index randomly generated spectrum indexes and as columns the indexes
        of the vector is converted to pickle.
        """
        assert not os.path.exists(self.s2v_embeddings_file_name), \
            "Given s2v_embeddings_file_name already exists"
        assert self.s2v_model is not None, "No spec2vec model was specified"
        # Convert Spectrum objects to SpectrumDocument
        spectrum_documents = create_spectrum_documents(
            self.list_of_spectra,
            progress_bar=self.progress_bars)
        embeddings_dict = {}
        for spectrum_id, spectrum_document in tqdm(enumerate(spectrum_documents),
                                                   desc="Calculating embeddings",
                                                   disable=not self.progress_bars):
            embedding = calc_vector(self.s2v_model,
                                    spectrum_document,
                                    allowed_missing_percentage=100)
            embeddings_dict[spectrum_id] = embedding

        # Convert to pandas Dataframe
        embeddings_dataframe = pd.DataFrame.from_dict(embeddings_dict,
                                                      orient="index")
        embeddings_dataframe.to_pickle(self.s2v_embeddings_file_name)
