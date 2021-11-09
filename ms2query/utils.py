import os
from typing import Union, List
import pickle
import pandas as pd
import numpy as np
from spec2vec.Spec2Vec import Spectrum
from matchms import importing


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def convert_files_to_matchms_spectrum_objects(file_name
                                              ) -> Union[List[Spectrum], None]:
    """Loads spectra from your spectrum file into memory as matchms Spectrum object

    The following file extensions can be loaded in with this function:
    "mzML", "json", "mgf", "msp", "mzxml", "usi" and "pickle".
    A pickled file is expected to directly contain a list of matchms spectrum objects.

    Args:
    -----
    file_name:
        Path to file containing spectra, with file extension "mzML", "json", "mgf", "msp",
        "mzxml", "usi" or "pickle"
    """
    assert os.path.exists(file_name), f"The specified file: {file_name} does not exists"

    file_extension = os.path.splitext(file_name)[1].lower()
    if file_extension == ".mzml":
        return list(importing.load_from_mzml(file_name))
    if file_extension == ".json":
        return list(importing.load_from_json(file_name))
    if file_extension == ".mgf":
        return list(importing.load_from_mgf(file_name))
    if file_extension == ".msp":
        return list(importing.load_from_msp(file_name))
    if file_extension == ".mzxml":
        return list(importing.load_from_mzxml(file_name))
    if file_extension == ".usi":
        return list(importing.load_from_usi(file_name))
    if file_extension == ".pickle":
        return load_pickled_file(file_name)
    print(f"File extension of file: {file_name} is not recognized")
    return None


def add_unknown_charges_to_spectra(spectrum_list: List[Spectrum],
                                   charge_to_use: int = 1) -> List[Spectrum]:
    """Adds charges to spectra when no charge is known

    This is important for matchms to be able to calculate the parent_mass
    from the mz_precursor

    Args:
    ------
    spectrum_list:
        List of spectra
    charge_to_use:
        The charge set when no charge is known. Default = 1
    """
    for spectrum in spectrum_list:
        if spectrum.get("charge") is None:
            spectrum.set("charge", charge_to_use)
    return spectrum_list


def get_classifier_from_csv_file(classifier_file_name: str,
                                 list_of_inchikeys: List[str]):
    """Returns a dataframe with the classifiers for a selection of inchikeys

    Args:
    ------
    csv_file_name:
        File name of text file with tap separated columns, with classifier
        information.
    list_of_inchikeys:
        list with the first 14 letters of inchikeys, that are selected from
        the classifier file.
    """
    assert os.path.isfile(classifier_file_name), \
        f"The given classifier csv file does not exist: {classifier_file_name}"
    classifiers_df = pd.read_csv(classifier_file_name, sep="\t")
    columns_to_keep = ["inchi_key", "smiles", "cf_kingdom",
                       "cf_superclass", "cf_class", "cf_subclass",
                       "cf_direct_parent", "npc_class_results",
                       "npc_superclass_results", "npc_pathway_results"]
    list_of_classifiers = []
    for inchikey in list_of_inchikeys:
        classifiers = classifiers_df.loc[
            classifiers_df["inchi_key"].str.startswith(inchikey)]  # pylint: disable=unsubscriptable-object
        if classifiers.empty:
            list_of_classifiers.append(pd.DataFrame(np.array(
                [[inchikey] + [np.nan] * (len(columns_to_keep) - 1)]),
                columns=columns_to_keep))
        else:
            classifiers = classifiers[columns_to_keep].iloc[:1]

            list_of_classifiers.append(classifiers)
    if len(list_of_classifiers) == 0:
        results = pd.DataFrame(columns=columns_to_keep)
    else:
        results = pd.concat(list_of_classifiers, axis=0, ignore_index=True)

    results["inchi_key"] = list_of_inchikeys
    results.rename(columns={"inchi_key": "inchikey"}, inplace=True)
    return results
