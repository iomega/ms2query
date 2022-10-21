from matchms import Spectrum
import numpy as np
from ms2query.create_new_library.clean_library_spectra import (clean_peaks_and_normalise_intensities_spectra,
                                                               clean_up_smiles_inchi_and_inchikeys,
                                                               remove_wrong_ion_modes,
                                                               remove_not_fully_annotated_spectra)


def test_clean_peaks_and_normalise_intensities_spectra(tmp_path):
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
    cleaned_spectra = clean_peaks_and_normalise_intensities_spectra(library_spectra)
    # Check if the spectra are still correct, output is not checked
    assert len(cleaned_spectra) == 2, "two spectra were expected after cleaning"
    assert isinstance(cleaned_spectra[0], Spectrum) and isinstance(cleaned_spectra[1], Spectrum), "Expected a list with two spectrum objects"


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
    cleaned_spectra = clean_up_smiles_inchi_and_inchikeys(library_spectra, True)
    # Check if the spectra are still correct, output is not checked
    assert len(cleaned_spectra) == 2, "two spectra were expected after cleaning"
    for i, spectrum in enumerate(cleaned_spectra):
        assert isinstance(spectrum, Spectrum), "Expected a list with spectra objects"
        assert spectrum.peaks == library_spectra[i].peaks, 'Expected that the peaks are not altered'

    assert cleaned_spectra[0].get("smiles") == "CCCC"
    assert cleaned_spectra[0].get("inchikey") == "IJDNQMDRQITEOD-UHFFFAOYSA-N"
    assert cleaned_spectra[0].get("inchi") == "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3"

    assert cleaned_spectra[1].get("smiles") == "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)CO)O)O)CO)O)O)O)O"
    assert cleaned_spectra[1].get("inchikey") == "CZMRCDWAGMRECN-UGDNZRGBSA-N"
    assert cleaned_spectra[1].get("inchi") == '"InChI=1S/C12H22O11/c13-1-4-6(16)8(18)9(19)11(21-4)23-12(3-15)10(20)7(17)5(2-14)22-12/h4-11,13-20H,1-3H2/t4-,5-,6-,7-,8+,9-,10+,11-,12+/m1/s1"'


def test_remove_not_fully_annotated_spectra(tmp_path):
    spectrum1 = Spectrum(
        mz=np.array([808.27356, 872.289917, 890.246277, 891.272888, 894.326416, 904.195679,
                     905.224548, 908.183472, 922.178101, 923.155762], dtype="float"),
        intensities=np.array([0.11106008, 0.12347332, 0.16352988, 0.17101522, 0.17312992, 0.19262333, 0.21442898,
                              0.42173288, 0.51071955, 1.], dtype="float"),
        metadata={'pepmass': (907.0, None), 'spectrumid': 'CCMSLIB00000001760', 'precursor_mz': 907.0,
                  'smiles': 'CCCC', "inchikey": "AAAAAAAAAAAAAAAA", "inchi": "this is a test inci", 'ionmode': "positive", "charge": 1})
    spectrum2 = Spectrum(
        mz=np.array([538.003174, 539.217773], dtype="float"),
        intensities=np.array([0.28046377, 0.28900242], dtype="float"),
        metadata={'pepmass': (928.0, None), 'spectrumid': 'CCMSLIB00000001761', 'precursor_mz': 342.30,
                  'compound_name': 'sucrose', "ionmode": "positive"})
    library_spectra = [spectrum1, spectrum2]
    results = remove_not_fully_annotated_spectra(library_spectra)
    assert len(results) == 1, "Expected that 1 spectrum was removed"
    assert spectrum1.__eq__(results[0]), "Expected an unaltered spectra"


if __name__ == "__main__":
    pass
