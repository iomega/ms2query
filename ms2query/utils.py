import os
import sys
import json
from typing import List, Union
import numpy as np
import pandas as pd
from matchms import importing
from spec2vec.Spec2Vec import Spectrum


if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


def load_ms2query_model(ms2query_model_file_name):
    """Loads in a MS2Query model

    a .pickle file is loaded like a ranadom forest from sklearn

    ms2query_model_file_name:
        The file name of the ms2query model
    """
    assert os.path.exists(ms2query_model_file_name), "MS2Query model file name does not exist"
    file_extension = os.path.splitext(ms2query_model_file_name)[1].lower()

    if file_extension == ".pickle":
        return load_pickled_file(ms2query_model_file_name)

    raise ValueError("The MS2Query model file is expected to end on .pickle")


def save_pickled_file(obj, filename: str):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def save_json_file(data, filename):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def load_json_file(filename):
    with open(filename, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def load_matchms_spectrum_objects_from_file(file_name
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
        spectra = load_pickled_file(file_name)
        assert isinstance(spectra, list), "Expected list of spectra"
        assert isinstance(spectra[0], Spectrum), "Expected list of spectra"
        return spectra
    assert False, f"File extension of file: {file_name} is not recognized"


def add_unknown_charges_to_spectra(spectrum_list: List[Spectrum],
                                   charge_to_use: int = 1,
                                   change_all_spectra: bool = False) -> List[Spectrum]:
    """Adds charges to spectra when no charge is known

    The charge is important to calculate the parent mass from the mz_precursor
    This function is not important anymore, since the switch to using mz_precursor

    Args:
    ------
    spectrum_list:
        List of spectra
    charge_to_use:
        The charge set when no charge is known. Default = 1
    change_all_spectra:
        If True the charge of all spectra is set to this value. If False only the spectra that do not have a specified
        charge will be changed.
    """
    if change_all_spectra:
        for spectrum in spectrum_list:
            spectrum.set("charge", charge_to_use)
    else:
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
    classifiers_df.rename(columns={"inchi_key": "inchikey"}, inplace=True)
    columns_to_keep = ["inchikey"] + column_names_for_output(False, True)
    list_of_classifiers = []
    for inchikey in list_of_inchikeys:
        classifiers = classifiers_df.loc[
            classifiers_df["inchikey"].str.startswith(inchikey)]
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

    results["inchikey"] = list_of_inchikeys
    return results


def column_names_for_output(return_non_classifier_columns: bool,
                            return_classifier_columns: bool,
                            additional_metadata_columns: List[str] = None,
                            additional_ms2query_score_columns: List[str] = None) -> List[str]:
    """Returns the column names for the output of results table

    This is used by the functions MS2Library.analog_search_store_in_csv, ResultsTable.export_to_dataframe
    and get_classifier_from_csv_file. The column names are used to select which data is added from the ResultsTable to
    the dataframe and the order of these columns is also used as order for the columns in this dataframe.

    Args:
    ------
    return_standard_columns:
        If true all columns are returned that do not belong to the classifier_columns. This always includes the
        standard_columns and if if additional_metadata_columns or additional_ms2query_score_columns is specified these
        are appended.
        If return_classifier_columns is True, the classifier_columns are also appended to the columns list.
    return_classifier_columns:
        If true the classifier columns appended. If return_standard_columns is false and return_classifier_columns is
        True, only the classifier columns are returned.
    additional_metadata_columns:
        These columns are appended to the standard columns and returned when return_non_classifier_columns is true
    additional_ms2query_score_columns:
        These columns are appended to the standard columns and returned when return_non_classifier_columns is true
    """
    standard_columns = ["query_spectrum_nr", "ms2query_model_prediction", "precursor_mz_difference", "precursor_mz_query_spectrum",
                        "precursor_mz_analog", "inchikey", "spectrum_ids", "analog_compound_name"]
    if additional_metadata_columns is not None:
        standard_columns += additional_metadata_columns
    if additional_ms2query_score_columns is not None:
        standard_columns += additional_ms2query_score_columns
    classifier_columns = ["smiles", "cf_kingdom", "cf_superclass", "cf_class", "cf_subclass",
                          "cf_direct_parent", "npc_class_results", "npc_superclass_results", "npc_pathway_results"]
    if return_classifier_columns and return_non_classifier_columns:
        return standard_columns + classifier_columns
    if return_classifier_columns:
        return classifier_columns
    if return_non_classifier_columns:
        return standard_columns
    return []
