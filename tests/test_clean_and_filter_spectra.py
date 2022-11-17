import os
import sys
import numpy as np
from matchms import Spectrum
from spec2vec import SpectrumDocument
from ms2query.clean_and_filter_spectra import (create_spectrum_documents,
                                               normalize_and_filter_peaks_multiple_spectra,
                                               normalize_and_filter_peaks,
                                               remove_wrong_ion_modes,
                                               harmonize_annotation,
                                               split_annotated_spectra,
                                               clean_normalize_and_split_annotated_spectra)

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


def test_minimal_processing_multiple_spectra():
    spectrum_1 = Spectrum(mz=np.array([5, 110, 220, 330, 399, 440],
                                       dtype="float"),
                          intensities=np.array([10, 10, 1, 10, 20, 100],
                                               dtype="float"),
                          metadata={"precursor_mz": 240.0})

    spectrum_2 = Spectrum(mz=np.array([110, 220, 330], dtype="float"),
                          intensities=np.array([0, 1, 10], dtype="float"),
                          metadata={"precursor_mz": 240.0}
                          )
    spectrum_list = [spectrum_1, spectrum_2]
    processed_spectrum_list = normalize_and_filter_peaks_multiple_spectra(spectrum_list)
    assert len(processed_spectrum_list) == 1, \
        "Expected only 1 spectrum, since spectrum 2 does not have enough peaks"
    found_spectrum = processed_spectrum_list[0]
    assert np.all(found_spectrum.peaks.mz == spectrum_1.peaks.mz), \
        "Expected different m/z values"
    assert np.all(found_spectrum.peaks.intensities ==
                  np.array([0.1, 0.1, 0.01, 0.1, 0.2, 1.])),\
        "Expected different intensities"


def test_normalize_and_filter_peaks_return_none():
    spectrum_in = Spectrum(mz=np.array([110, 220], dtype="float"),
                           intensities=np.array([10, 1],
                                                dtype="float"))
    spectrum = normalize_and_filter_peaks(spectrum_in)

    assert spectrum is None, \
        "Expected None because the number of peaks <3"


def test_normalize_and_filter_peaks(tmp_path):
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
    cleaned_spectra = [normalize_and_filter_peaks(s) for s in library_spectra]
    cleaned_spectra = [spectrum for spectrum in cleaned_spectra if spectrum is not None]
    # Check if the spectra are still correct, output is not checked
    assert len(cleaned_spectra) == 2, "two spectra were expected after cleaning"
    assert isinstance(cleaned_spectra[0], Spectrum) and isinstance(cleaned_spectra[1], Spectrum), "Expected a list with two spectrum objects"


def test_create_spectrum_documents():
    path_to_pickled_file = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/first_10_spectra.pickle')
    with open(path_to_pickled_file, "rb") as pickled_file:
        spectrum_list = pickle.load(pickled_file)
    spectrum_list = normalize_and_filter_peaks_multiple_spectra(spectrum_list)

    spectrum_documents = create_spectrum_documents(spectrum_list)
    assert isinstance(spectrum_documents, list), \
        "A list with spectrum_documents is expected"
    for spectrum_doc in spectrum_documents:
        assert isinstance(spectrum_doc, SpectrumDocument), \
            "A list with spectrum_documents is expected"


def test_clean_up_smiles_inchi_and_inchikeys(tmp_path):
    spectrum1 = Spectrum(
        mz=np.array([808.27356, 872.289917, 890.246277, 891.272888, 894.326416, 904.195679,
                     905.224548, 908.183472, 922.178101, 923.155762], dtype="float"),
        intensities=np.array([0.11106008, 0.12347332, 0.16352988, 0.17101522, 0.17312992, 0.19262333, 0.21442898,
                              0.42173288, 0.51071955, 1.], dtype="float"),
        metadata={'pepmass': (907.0, None), 'spectrumid': 'CCMSLIB00000001760', 'precursor_mz': 907.0,
                  'smiles': 'CCCC', 'ionmode': "positive"})
    spectrum2 = Spectrum(
        mz=np.array([538.003174, 539.217773], dtype="float"),
        intensities=np.array([0.28046377, 0.28900242], dtype="float"),
        metadata={'pepmass': (928.0, None), 'spectrumid': 'CCMSLIB00000001761', 'precursor_mz': 342.30,
                  'compound_name': 'sucrose', "ionmode": "positive"})
    library_spectra = [spectrum1, spectrum2]
    cleaned_spectrum_1 = harmonize_annotation(spectrum1, True)
    cleaned_spectrum_2 = harmonize_annotation(spectrum2, True)

    assert isinstance(cleaned_spectrum_1, Spectrum), "Expected a list with spectra objects"
    assert isinstance(cleaned_spectrum_2, Spectrum), "Expected a list with spectra objects"
    assert cleaned_spectrum_1.peaks == library_spectra[0].peaks, 'Expected that the peaks are not altered'
    assert cleaned_spectrum_2.peaks == library_spectra[1].peaks, 'Expected that the peaks are not altered'

    assert cleaned_spectrum_1.get("smiles") == "CCCC"
    assert cleaned_spectrum_1.get("inchikey") == "IJDNQMDRQITEOD-UHFFFAOYSA-N"
    assert cleaned_spectrum_1.get("inchi") == "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3"

    assert cleaned_spectrum_2.get("smiles") == "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)CO)O)O)CO)O)O)O)O"
    assert cleaned_spectrum_2.get("inchikey") == "CZMRCDWAGMRECN-UGDNZRGBSA-N"
    assert cleaned_spectrum_2.get("inchi") == '"InChI=1S/C12H22O11/c13-1-4-6(16)8(18)9(19)11(21-4)23-12(3-15)10(20)7(17)5(2-14)22-12/h4-11,13-20H,1-3H2/t4-,5-,6-,7-,8+,9-,10+,11-,12+/m1/s1"'


def test_remove_not_fully_annotated_spectra(tmp_path):
    spectrum1 = Spectrum(
        mz=np.array([808.27356, 872.289917, 890.246277, 891.272888, 894.326416, 904.195679,
                     905.224548, 908.183472, 922.178101, 923.155762], dtype="float"),
        intensities=np.array([0.11106008, 0.12347332, 0.16352988, 0.17101522, 0.17312992, 0.19262333, 0.21442898,
                              0.42173288, 0.51071955, 1.], dtype="float"),
        metadata={'pepmass': (907.0, None), 'spectrumid': 'CCMSLIB00000001760', 'precursor_mz': 907.0,
                  'smiles': 'CCCC', "inchikey": "CZMRCDWAGMRECN-UGDNZRGBSA-N", "inchi": "InChI=1S/C12H22O11/c13-1-4-6(16)8(18)9(19)11(21-4)23-12(3-15)10(20)7(17)5(2-14)22-12/h4-11,13-20H,1-3H2/t4-,5-,6-,7-,8+,9-,10+,11-,12+/m1/s1", 'ionmode': "positive", "charge": 1})
    spectrum2 = Spectrum(
        mz=np.array([538.003174, 539.217773], dtype="float"),
        intensities=np.array([0.28046377, 0.28900242], dtype="float"),
        metadata={'pepmass': (928.0, None), 'spectrumid': 'CCMSLIB00000001761', 'precursor_mz': 342.30,
                  'compound_name': 'sucrose', "ionmode": "positive"})
    library_spectra = [spectrum1, spectrum2]
    results = split_annotated_spectra(library_spectra)
    assert len(results[0]) == 1, "Expected that 1 spectrum was removed"
    assert spectrum1.__eq__(results[0][0]), "Expected an unaltered spectra"
    assert len(results[0]) == 1, "Expected that 1 spectrum was removed"
    assert spectrum2.__eq__(results[1][0]), "Expected an unaltered spectra"


def test_remove_wrong_ion_mode():
    spectrum1 = Spectrum(
        mz=np.array([808.27356], dtype="float"),
        intensities=np.array([0.11106008], dtype="float"),
        metadata={'pepmass': (907.0, None), 'spectrumid': 'CCMSLIB00000001760', 'precursor_mz': 907.0,
                   'ionmode': "positive", "charge": 1})
    spectrum2 = Spectrum(
        mz=np.array([538.003174, 539.217773], dtype="float"),
        intensities=np.array([0.28046377, 0.28900242], dtype="float"),
        metadata={'pepmass': (928.0, None), 'spectrumid': 'CCMSLIB00000001761', 'precursor_mz': 342.30,
                  'compound_name': 'sucrose', "ionmode": "negative"})
    result = remove_wrong_ion_modes([spectrum1, spectrum2], "positive")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == spectrum1


def test_preprocess_library_spectra():
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
    cleaned_spectra = clean_normalize_and_split_annotated_spectra([spectrum1, spectrum2], "positive")[0]
    for spectrum in cleaned_spectra:
        assert isinstance(spectrum, Spectrum)


if __name__ == "__main__":
    pass
