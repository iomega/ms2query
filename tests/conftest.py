import os
import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from matchms.importing.load_from_mgf import load_from_mgf
from ms2query.ms2library import MS2Library
from ms2query.query_from_sqlite_database import SqliteLibrary
from ms2query.utils import load_df_from_parquet_file


@pytest.fixture(scope="package")
def path_to_general_test_files() -> str:
    return os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files')


@pytest.fixture(scope="package")
def sqlite_library(path_to_general_test_files):
    path_to_library = os.path.join(path_to_general_test_files, "100_test_spectra.sqlite")
    return SqliteLibrary(path_to_library)


@pytest.fixture(scope="package")
def ms2library(path_to_general_test_files) -> MS2Library:
    """Returns file names of the files needed to create MS2Library object"""
    sqlite_file_loc = os.path.join(
        path_to_general_test_files,
        "100_test_spectra.sqlite")
    spec2vec_model_file_loc = os.path.join(
        path_to_general_test_files,
        "100_test_spectra_s2v_model.model")
    s2v_pickled_embeddings_file = os.path.join(
        path_to_general_test_files,
        "100_test_spectra_s2v_embeddings.parquet")
    ms2ds_model_file_name = os.path.join(
        path_to_general_test_files,
        "ms2deepscore_model.pt")
    ms2ds_embeddings_file_name = os.path.join(
        path_to_general_test_files,
        "100_test_spectra_ms2ds_embeddings.parquet")
    ms2q_model_file_name = os.path.join(path_to_general_test_files,
        "test_ms2q_rf_model.onnx")
    ms2library = MS2Library(sqlite_file_loc, spec2vec_model_file_loc, ms2ds_model_file_name,
                            s2v_pickled_embeddings_file, ms2ds_embeddings_file_name, ms2q_model_file_name)
    return ms2library


@pytest.fixture(scope="package")
def test_spectra():
    """Returns a list with two spectra

    The spectra are created by using peaks from the first two spectra in
    100_test_spectra.pickle, to make sure that the peaks occur in the s2v
    model. The other values are random.
    """
    spectrum1 = Spectrum(mz=np.array([808.27356, 872.289917, 890.246277,
                                      891.272888, 894.326416, 904.195679,
                                      905.224548, 908.183472, 922.178101,
                                      923.155762], dtype="float"),
                         intensities=np.array([0.11106008, 0.12347332,
                                               0.16352988, 0.17101522,
                                               0.17312992, 0.19262333,
                                               0.21442898, 0.42173288,
                                               0.51071955, 1.],
                                              dtype="float"),
                         metadata={'pepmass': (907.0, None),
                                   'spectrumid': 'CCMSLIB00000001760',
                                   'precursor_mz': 907.0,
                                   # 'precursor_mz': 905.9927235480093,
                                   'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N',
                                   'charge': 1,
                                   "ionmode": "positive"})
    spectrum2 = Spectrum(mz=np.array([538.003174, 539.217773, 556.030396,
                                      599.352783, 851.380859, 852.370605,
                                      909.424438, 953.396606, 963.686768,
                                      964.524658
                                      ],
                                     dtype="float"),
                         intensities=np.array([0.28046377, 0.28900242,
                                               0.31933114, 0.32199162,
                                               0.34214536, 0.35616456,
                                               0.36216307, 0.41616014,
                                               0.71323034, 1.],
                                              dtype="float"),
                         metadata={'pepmass': (928.0, None),
                                   'spectrumid': 'CCMSLIB00000001761',
                                   'precursor_mz': 928.0,
                                   # 'precursor_mz': 905.010782,
                                   'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N',
                                   # 'charge': 1
                                   })
    spectra = [spectrum1, spectrum2]
    return spectra


@pytest.fixture(scope="package")
def hundred_test_spectra(path_to_general_test_files):
    return list(load_from_mgf(os.path.join(path_to_general_test_files, "100_test_spectra.mgf"),
                metadata_harmonization=True))


@pytest.fixture(scope="package")
def expected_tanimoto_scores_df(path_to_general_test_files):
    return pd.read_csv(os.path.join(path_to_general_test_files,
                                    "tanimoto_scores_100_test_spectra.csv"), index_col=0)


@pytest.fixture(scope="package")
def expected_ms2ds_embeddings(path_to_general_test_files):
    expected_embeddings = load_df_from_parquet_file(os.path.join(
        path_to_general_test_files,
        "100_test_spectra_ms2ds_embeddings.parquet"))
    return expected_embeddings


@pytest.fixture(scope="package")
def expected_s2v_embeddings(path_to_general_test_files):
    expected_embeddings = load_df_from_parquet_file(os.path.join(
        path_to_general_test_files,
        "100_test_spectra_s2v_embeddings.parquet"))
    return expected_embeddings
