import os
import pytest
import numpy as np
from matchms import Spectrum
from ms2query.ms2library import MS2Library
from ms2query.benchmarking.collect_test_data_results import generate_test_results_ms2query
from tests.test_ms2library import ms2library


@pytest.fixture
def test_spectra():
    """Returns a list with two spectra

    The spectra are created by using peaks from the first two spectra in
    100_test_spectra.pickle, to make sure that the peaks occur in the s2v
    model. The other values are random.
    """
    spectrum1 = Spectrum(
        mz=np.array([808.27356, 872.289917, 890.246277, 891.272888, 894.326416, 904.195679], dtype="float"),
        intensities=np.array([0.11106008, 0.12347332, 0.16352988, 0.17101522, 0.17312992, 1.], dtype="float"),
        metadata={'pepmass': (907.0, None),
                  'spectrumid': 'CCMSLIB00000001760',
                  'precursor_mz': 907.0,
                  'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N',
                  'smiles': "CCCC",
                  'charge': 1})
    spectrum2 = Spectrum(
        mz=np.array([538.003174, 539.217773, 556.030396, 599.352783, 851.380859, 852.370605], dtype="float"),
        intensities=np.array([0.28046377, 0.28900242, 0.31933114, 0.32199162, 0.71323034, 1.], dtype="float"),
        metadata={'pepmass': (928.0, None),
                  'spectrumid': 'CCMSLIB00000001761',
                  'precursor_mz': 928.0,
                  'inchikey': 'SCYRNRIZFGMUSB-STOGWRBBSA-N',
                  'smiles': "CCCCC"
                  })
    return [spectrum1, spectrum2]


def test_generate_test_data(ms2library, test_spectra, tmp_path):
    result = generate_test_results_ms2query(ms2library,
                                            test_spectra,
                                            os.path.join(tmp_path, "temporary.csv"))
    assert result[0] == ['CCMSLIB00000001548', 0.5645, 0.003861003861003861, test_spectra[0]]
    assert result[1] == ['CCMSLIB00000001660', 0.409, 0.010610079575596816, test_spectra[1]]
