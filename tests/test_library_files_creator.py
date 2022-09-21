import os
import pandas as pd
import numpy as np
import pytest
from matchms import Spectrum
from ms2query.library_files_creator import LibraryFilesCreator
from ms2query.utils import load_pickled_file, convert_files_to_matchms_spectrum_objects


@pytest.fixture
def path_to_general_test_files() -> str:
    return os.path.join(
        os.getcwd(),
        'tests/test_files/general_test_files')

def test_set_settings_correct(path_to_general_test_files):
    """Tests if settings are set correctly"""
    library_spectra = convert_files_to_matchms_spectrum_objects(os.path.join(
        path_to_general_test_files, '100_test_spectra.pickle'))
    test_create_files = LibraryFilesCreator(library_spectra,
                                            output_base_filename="test_output_name",
                                            progress_bars=False)
    assert test_create_files.settings["output_file_sqlite"] == \
           "test_output_name.sqlite", "Expected different output_file_sqlite"
    assert test_create_files.settings["progress_bars"] is False, \
           "Expected different setting for progress_bar"
    assert test_create_files.settings["spectrum_id_column_name"] == \
           "spectrumid", "Expected different spectrum_id_column_name"
    assert test_create_files.settings["ms2ds_embeddings_file_name"] == \
           "test_output_name_ms2ds_embeddings.pickle", \
           "Expected different ms2ds_embeddings_file_name"
    assert test_create_files.settings["s2v_embeddings_file_name"] == \
           "test_output_name_s2v_embeddings.pickle", \
           "Expected different s2v_embeddings_file_name"


def test_set_settings_wrong():
    """Tests if an error is raised if a wrong attribute is passed"""
    pickled_spectra_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/100_test_spectra.pickle')
    pytest.raises(AssertionError, LibraryFilesCreator,
                  pickled_spectra_file_name, "output_filename",
                  not_recognized_attribute="test_value")


def test_give_already_used_file_name(tmp_path):
    base_file_name = os.path.join(tmp_path, "base_file_name")
    already_existing_file = base_file_name + ".sqlite"
    with open(already_existing_file, "w") as file:
        file.write("test")

    pickled_spectra_file_name = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/test_files_ms2library/100_test_spectra.pickle')
    pytest.raises(AssertionError, LibraryFilesCreator,
                  pickled_spectra_file_name, base_file_name)


def test_store_ms2ds_embeddings(tmp_path, path_to_general_test_files):
    """Tests store_ms2ds_embeddings"""
    base_file_name = os.path.join(tmp_path, '100_test_spectra')
    library_spectra = convert_files_to_matchms_spectrum_objects(os.path.join(
        path_to_general_test_files, '100_test_spectra.pickle'))
    test_create_files = LibraryFilesCreator(library_spectra, base_file_name,
        ms2ds_model_file_name=os.path.join(path_to_general_test_files, 'ms2ds_siamese_210301_5000_500_400.hdf5'))
    test_create_files.clean_peaks_and_normalise_intensities_spectra()
    test_create_files.store_ms2ds_embeddings()

    new_embeddings_file_name = base_file_name + "_ms2ds_embeddings.pickle"
    assert os.path.isfile(new_embeddings_file_name), \
        "Expected file to be created"
    # Test if correct embeddings are stored
    embeddings = load_pickled_file(new_embeddings_file_name)
    expected_embeddings = load_pickled_file(os.path.join(
        path_to_general_test_files,
        "test_files_without_spectrum_id",
        "100_test_spectra_ms2ds_embeddings.pickle"))
    pd.testing.assert_frame_equal(embeddings, expected_embeddings,
                                  check_exact=False,
                                  atol=1e-5)


def test_store_s2v_embeddings(tmp_path, path_to_general_test_files):
    """Tests store_ms2ds_embeddings"""
    base_file_name = os.path.join(tmp_path, '100_test_spectra')
    library_spectra = convert_files_to_matchms_spectrum_objects(os.path.join(
        path_to_general_test_files, '100_test_spectra.pickle'))
    test_create_files = LibraryFilesCreator(library_spectra, base_file_name,
        s2v_model_file_name=os.path.join(path_to_general_test_files, "100_test_spectra_s2v_model.model"))
    test_create_files.clean_peaks_and_normalise_intensities_spectra()
    test_create_files.store_s2v_embeddings()

    new_embeddings_file_name = base_file_name + "_s2v_embeddings.pickle"
    assert os.path.isfile(new_embeddings_file_name), \
        "Expected file to be created"
    embeddings = load_pickled_file(new_embeddings_file_name)
    expected_embeddings = load_pickled_file(os.path.join(
        path_to_general_test_files,
        "test_files_without_spectrum_id",
        "100_test_spectra_s2v_embeddings.pickle"))
    pd.testing.assert_frame_equal(embeddings, expected_embeddings,
                                  check_exact=False,
                                  atol=1e-5)


def test_calculate_tanimoto_scores(tmp_path, path_to_general_test_files):
    base_file_name = os.path.join(tmp_path, '100_test_spectra')
    library_spectra = convert_files_to_matchms_spectrum_objects(os.path.join(
        path_to_general_test_files, '100_test_spectra.pickle'))
    test_create_files = LibraryFilesCreator(library_spectra,
                                            base_file_name)
    test_create_files.calculate_tanimoto_scores()
    result: pd.DataFrame = test_create_files.tanimoto_scores
    result.sort_index(inplace=True)
    result.sort_index(1, inplace=True)
    expected_result = load_pickled_file(path_to_general_test_files + "/100_test_spectra_tanimoto_scores.pickle")
    pd.testing.assert_frame_equal(result, expected_result, check_exact=False, atol=1e-5)


def test_clean_library_spectra(tmp_path, path_to_general_test_files):
    base_file_name = os.path.join(tmp_path, '100_test_spectra')

    spectrum1 = Spectrum(
        mz=np.array([808.27356, 872.289917, 890.246277, 891.272888, 894.326416, 904.195679,
                     905.224548, 908.183472, 922.178101, 923.155762], dtype="float"),
        intensities=np.array([0.11106008, 0.12347332, 0.16352988, 0.17101522, 0.17312992, 0.19262333, 0.21442898,
                              0.42173288, 0.51071955, 1.], dtype="float"),
        metadata={'pepmass': (907.0, None), 'spectrumid': 'CCMSLIB00000001760', 'precursor_mz': 907.0,
                  'smiles': 'CCCC', 'ionmode': "positive"})
    spectrum2 = Spectrum(
        mz=np.array([538.003174, 539.217773, 556.030396, 599.352783, 851.380859, 852.370605, 909.424438, 953.396606,
                     963.686768, 964.524658], dtype="float"),
        intensities=np.array([0.28046377, 0.28900242, 0.31933114, 0.32199162, 0.34214536, 0.35616456, 0.36216307,
                              0.41616014, 0.71323034, 1.], dtype="float"),
        metadata={'pepmass': (928.0, None), 'spectrumid': 'CCMSLIB00000001761', 'precursor_mz': 342.30,
                  'compound_name': 'sucrose', "ionmode": "positive"})
    library_spectra = [spectrum1, spectrum2]
    test_create_files = LibraryFilesCreator(library_spectra,
                                            base_file_name,
                                            ion_mode="positive")
    test_create_files.clean_peaks_and_normalise_intensities_spectra()
    test_create_files.clean_up_smiles_inchi_and_inchikeys(True)
    test_create_files.remove_not_fully_annotated_spectra()
    filtered_spectra = test_create_files.list_of_spectra
    print(filtered_spectra[0].metadata)
    print(filtered_spectra[1].metadata)
    assert False