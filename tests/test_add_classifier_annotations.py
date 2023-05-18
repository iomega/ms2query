import numpy as np
from matchms import Spectrum
import pytest
from ms2query.create_new_library.add_classifire_classifications import select_compound_classes


@pytest.fixture
def spectra():
    spectrum1 = Spectrum(mz=np.array([100], dtype="float"),
                         intensities=np.array([1.], dtype="float"),
                         metadata={'inchikey': 'WXDBUBIFYCCNLE-NSCMQRKRSA-N',
                                   'smiles': 'CCCCCCC[C@@H](C/C=C/CCC(=O)NC/C(=C/Cl)/[C@@]12[C@@H](O1)[C@H](CCC2=O)O)OC',
                                   'charge': 1})
    spectrum2 = Spectrum(mz=np.array([538.003174], dtype="float"),
                         intensities=np.array([0.28046377], dtype="float"),
                         metadata={'inchikey': 'WRIPSIKIDAUKBP-MDWZMJQESA-N',
                                   'smiles': 'CC1CCCC(=O)N(/C=C/CCC(C(=O)OC(C1)C(C)(C)C)C)C'})
    return [spectrum1, spectrum2]


def test_add_classifier_annotation(spectra):
    result = select_compound_classes(spectra)
    assert sorted(result) == [['WRIPSIKIDAUKBP', 'Organic compounds', 'Phenylpropanoids and polyketides', 'Macrolactams', '', 'Macrolactams', '', '', 'Alkaloids', 'False'],
                              ['WXDBUBIFYCCNLE', 'Organic compounds', 'Organoheterocyclic compounds', 'Oxepanes', '', 'Oxepanes', 'Lipopeptides', 'Oligopeptides', 'Amino acids and Peptides', 'False']]
