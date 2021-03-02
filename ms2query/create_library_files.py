from typing import List
import pandas as pd
import os
from tqdm import tqdm
from matchms.Spectrum import Spectrum
from gensim.models import Word2Vec
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import SiameseModel
from ms2deepscore.models import load_model as load_ms2ds_model
from spec2vec.vector_operations import calc_vector
from ms2query.ms2library import create_spectrum_documents
from ms2query.create_sqlite_database import make_sqlfile_wrapper
from ms2query.spectrum_processing import minimal_processing_multiple_spectra
from ms2query.app_helpers import load_pickled_file


def create_all_library_files(pickled_spectra_file_name: str,
                             ms2ds_model_file_name: str,
                             s2v_model_file_name: str,
                             new_sqlite_file_name,
                             new_ms2ds_embeddings_file_name,
                             new_s2v_embeddings_file_name,
                             tanimoto_scores_file_name: str,
                             progress_bars: bool = True,
                             spectrum_column_name: str = 'spectrumid',
                             calculate_new_tanimoto_scores: bool = False):

    # Loads the spectra from a pickled file
    list_of_spectra = load_pickled_file(pickled_spectra_file_name)
    assert list_of_spectra[0].get(spectrum_column_name), \
        f"Expected spectra to have '{spectrum_column_name}' in metadata, " \
        "probably named 'spectrum_id' or 'spectrumid'"
    # # Does normalization and filtering of spectra
    list_of_spectra = \
        minimal_processing_multiple_spectra(list_of_spectra,
                                            progress_bar=progress_bars)
    if calculate_new_tanimoto_scores:
        assert os.path.exists(tanimoto_scores_file_name),\
            "Tanimoto scores file already exists, " \
            "to use a file with already calculated tanimoto scores, " \
            "set calculate_new_tanimoto_scores to False"
        # Todo automatically create tanimoto scores

    make_sqlfile_wrapper(new_sqlite_file_name,
                         tanimoto_scores_file_name,
                         list_of_spectra,
                         columns_dict={"parent_mass": "REAL"},
                         progress_bars=progress_bars,
                         spectrum_column_name=spectrum_column_name)

    store_s2v_embeddings(list_of_spectra,
                         s2v_model_file_name,
                         new_s2v_embeddings_file_name)

    store_ms2ds_embeddings(list_of_spectra,
                           ms2ds_model_file_name,
                           new_ms2ds_embeddings_file_name,
                           spectrum_column_name=spectrum_column_name,
                           progress_bar=progress_bars)


def store_ms2ds_embeddings(spectrum_list: List[Spectrum],
                           ms2ds_model_file_name: str,
                           new_pickled_embeddings_file_name: str,
                           spectrum_column_name: str = 'spectrumid',
                           progress_bar: bool = True):
    """Creates a pickled file with embeddings scores for spectra

    A dataframe with as index the spectrum_ids and as columns the indexes of
    the vector is converted to pickle.

    Args:
    ------
    spectrum_list:
        Spectra for which embeddings should be calculated.
    ms2ds_model_file_name:
        File name for a SiameseModel that is used to calculate the embeddings.
    new_picled_embeddings_file_name:
        The file name in which the pickled dataframe is stored.
    """
    model = load_ms2ds_model(ms2ds_model_file_name)
    ms2ds = MS2DeepScore(model)

    embeddings = []
    for spec in tqdm(spectrum_list,
                     desc="Calculating ms2ds embeddings",
                     disable= not progress_bar):
        binned_spec = model.spectrum_binner.transform([spec],
                                                      progress_bar=False)[0]
        embeddings.append(
            model.base.predict(ms2ds._create_input_vector(binned_spec))[0])

    spectra_vector_dataframe = pd.DataFrame(
        embeddings,
        index=[spectrum.get(spectrum_column_name) for spectrum in spectrum_list])
    spectra_vector_dataframe.to_pickle(new_pickled_embeddings_file_name)


def store_s2v_embeddings(spectra_list: List[Spectrum],
                         s2v_model_file_name: str,
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
    s2v_model_file_name:
        File name to load spec2vec model, 3 files are expected, with the
         extension .model, .model.trainables.syn1neg.npy and
         .model.wv.vectors.npy, together containing a Spec2Vec model,
    new_pickled_embeddings_file_name:
        File name for file created
    progress_bars:
        When True progress bars and steps in progress will be shown.
        Default = True
    """
    model = Word2Vec.load(s2v_model_file_name)
    # Convert Spectrum objects to SpectrumDocument
    spectrum_documents = create_spectrum_documents(spectra_list,
                                                   progress_bar=progress_bars)
    embeddings_dict = {}
    for spectrum_document in tqdm(spectrum_documents,
                                  desc="Calculating embeddings",
                                  disable=not progress_bars):
        embedding = calc_vector(model,
                                spectrum_document,
                                allowed_missing_percentage=100)
        embeddings_dict[spectrum_document.get("spectrumid")] = embedding

    # Convert to pandas Dataframe
    embeddings_dataframe = pd.DataFrame.from_dict(embeddings_dict,
                                                  orient="index")
    embeddings_dataframe.to_pickle(new_pickled_embeddings_file_name)


if __name__ == "__main__":
    ms2ds_model_file_name = "../../ms2deepscore/data/" \
            "ms2ds_siamese_210207_ALL_GNPS_positive_L1L2.hdf5"
    spectra_file = "../downloads/gnps_210125/spectra/spectra_gnps_210125_cleaned_parent_mass"
    s2v_model_file = "../downloads/" \
            "spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
    sqlite_file_name = "test_sqlite_file.sqlite"
    ms2ds_embeddings_file_name = "test_ms2ds_embeddings"
    s2v_embeddings_file_name = "test_s2v_embeddings"
    tanimoto_scores_file = "../tests/test_files/test_tanimoto_scores.pickle"

    create_all_library_files(spectra_file,
                             ms2ds_model_file_name,
                             s2v_model_file,
                             sqlite_file_name,
                             ms2ds_embeddings_file_name,
                             s2v_embeddings_file_name,
                             tanimoto_scores_file)
