import os
from ms2query.utils import load_matchms_spectrum_objects_from_file
from ms2query.create_new_library.split_data_for_training import split_spectra_on_inchikeys, \
    split_spectra_in_random_inchikey_sets, select_unique_inchikeys


def test_split_spectra_on_inchikeys(path_to_general_test_files, hundred_test_spectra):
    training_spectra, validation_spectra = split_spectra_on_inchikeys(hundred_test_spectra, 10)
    assert isinstance(training_spectra, list)
    assert isinstance(validation_spectra, list)
    assert len(select_unique_inchikeys(validation_spectra)) + len(select_unique_inchikeys(training_spectra)) == len(select_unique_inchikeys(hundred_test_spectra))
    assert len(select_unique_inchikeys(validation_spectra)) == len(select_unique_inchikeys(hundred_test_spectra))//10


def test_split_spectra_in_random_inchikey_sets(path_to_general_test_files, hundred_test_spectra):
    spectra_sets = split_spectra_in_random_inchikey_sets(hundred_test_spectra, 10)
    assert isinstance(spectra_sets, list), "Expected list"
    assert len(spectra_sets) == 10, "Expected 10 lists of spectra"
    nr_of_inchikeys = 0
    nr_of_spectra = 0
    for spectrum_set in spectra_sets:
        assert isinstance(spectrum_set, list), "expected list of list"
        inchikey_set = select_unique_inchikeys(spectrum_set)
        assert len(inchikey_set) in [6, 7], "Expected the number of inchikeys per set to be 6 or 7"
        nr_of_inchikeys += len(inchikey_set)
        nr_of_spectra += len(spectrum_set)
    assert nr_of_inchikeys == len(select_unique_inchikeys(hundred_test_spectra)), \
        "The sum of the number of inchikeys in each set, does not equal the total number of inchikeys"
    assert nr_of_spectra == len(hundred_test_spectra), \
        "The sum of the number of inchikeys in each set, does not equal the total number of inchikeys"
