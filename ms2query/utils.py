import json
import os
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from matchms import importing
from onnxruntime import InferenceSession
from spec2vec.Spec2Vec import Spectrum


def load_ms2query_model(ms2query_model_file_name) -> InferenceSession:
    """Loads in a MS2Query model

    a .pickle file is loaded like a ranadom forest from sklearn

    ms2query_model_file_name:
        The file name of the ms2query model
    """
    assert os.path.exists(ms2query_model_file_name), "MS2Query model file name does not exist"
    file_extension = os.path.splitext(ms2query_model_file_name)[1].lower()
    assert file_extension == ".onnx", \
        "An file in Onnx format is expected, older versions used a .pickle file for the random forest model. " \
        "Downgrade MS2Query to an older version or download the new model."
    if file_extension == ".onnx":
        return InferenceSession(ms2query_model_file_name)

    raise ValueError("The MS2Query model file is expected to end on .pickle")


def save_df_as_parquet_file(df, filename: str):
    assert not os.path.exists(filename), "File already exists"
    df.columns = df.columns.astype(str)
    df.to_parquet(filename)


def save_json_file(data, filename):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def load_json_file(filename):
    with open(filename, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_df_from_parquet_file(filename: str):
    return pd.read_parquet(filename)


def load_matchms_spectrum_objects_from_file(file_name
                                            ) -> Union[List[Spectrum], None]:
    """Loads spectra from your spectrum file into memory as matchms Spectrum object

    The following file extensions can be loaded in with this function:
    "mzML", "json", "mgf", "msp", "mzxml" and "usi".
    A pickled file is expected to directly contain a list of matchms spectrum objects.

    Args:
    -----
    file_name:
        Path to file containing spectra, with file extension "mzML", "json", "mgf", "msp",
        "mzxml" or "usi".
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


def column_names_for_output(return_non_classifier_columns: bool,
                            return_classifier_columns: bool,
                            additional_metadata_columns: Tuple[str, ...] = None,
                            additional_ms2query_score_columns: Tuple[str, ...] = None) -> List[str]:
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
                        "precursor_mz_analog", "inchikey", "analog_compound_name", "smiles",]
    if additional_metadata_columns is not None:
        standard_columns += additional_metadata_columns
    if additional_ms2query_score_columns is not None:
        standard_columns += additional_ms2query_score_columns
    classifier_columns = ["cf_kingdom", "cf_superclass", "cf_class", "cf_subclass",
                          "cf_direct_parent", "npc_class_results", "npc_superclass_results", "npc_pathway_results"]
    if return_classifier_columns and return_non_classifier_columns:
        return standard_columns + classifier_columns
    if return_classifier_columns:
        return classifier_columns
    if return_non_classifier_columns:
        return standard_columns
    return []


def return_non_existing_file_name(file_name):
    """Checks if a path already exists, otherwise creates a new filename with (1)"""
    if not os.path.exists(file_name):
        return file_name
    print(f"The file name already exists: {file_name}")
    file_name_base, file_extension = os.path.splitext(file_name)
    i = 1
    new_file_name = f"{file_name_base}({i}){file_extension}"
    while os.path.exists(new_file_name):
        i += 1
        new_file_name = f"{file_name_base}({i}){file_extension}"
    print(f"Instead the file will be stored in {new_file_name}")
    return new_file_name


class SettingsRunMS2Query:
    """
    preselection_cut_off:
            The number of spectra with the highest ms2ds that should be
            selected. Default = 2000
        nr_of_top_analogs_to_save:
            The number of returned analogs that are stored.
        minimal_ms2query_metascore:
            The minimal ms2query metascore needed to be stored in the csv file.
            Spectra for which no analog with this minimal metascore was found,
            will not be stored in the csv file.
        additional_metadata_columns:
            Additional columns with query spectrum metadata that should be added. For instance "retention_time".
        additional_ms2query_score_columns:
            Additional columns with scores used for calculating the ms2query metascore
            Options are: "s2v_score", "ms2ds_score", "average_ms2deepscore_multiple_library_structures",
            "average_tanimoto_score_library_structures"
    """
    def __init__(self,
                 nr_of_top_analogs_to_save: int = 1,
                 minimal_ms2query_metascore: int = 0,
                 additional_metadata_columns: tuple = ("retention_time", "retention_index",),
                 additional_ms2query_score_columns: tuple = (),
                 preselection_cut_off: int = 2000,
                 filter_on_ion_mode: Optional[str] = None,
                 ):
        # pylint: disable=too-many-arguments
        self.nr_of_top_analogs_to_save = nr_of_top_analogs_to_save
        self.minimal_ms2query_metascore = minimal_ms2query_metascore
        self.additional_metadata_columns = tuple(additional_column.lower()
                                                 for additional_column in additional_metadata_columns)
        self.additional_ms2query_score_columns = additional_ms2query_score_columns
        self.preselection_cut_off = preselection_cut_off
        self.filter_on_ion_mode = filter_on_ion_mode


def predict_onnx_model(random_forest_onnx_model: InferenceSession, input_values):
    """Makes predictions for an onnx model"""
    # input_name = random_forest_onnx_model.get_inputs()[0].name
    predictions = random_forest_onnx_model.run(None, {"input": input_values.astype(np.float32)})
    # Converts np array of shape (n,1) to a 1D numpy array.
    predictions = predictions[0].ravel()
    return predictions


def select_files_in_directory(directory):
    """Returns the file names in a directory"""
    assert os.path.exists(directory) \
           and not os.path.isfile(directory), "Expected a directory"
    # Select all files in the directory
    files_in_directory = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        # skip folders
        if os.path.isfile(file_path):
            files_in_directory.append(file_name)
    return files_in_directory
