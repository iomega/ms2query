import numpy as np
import pandas as pd
import re
from typing import Union, List
from matchms.Spectrum import Spectrum
from ms2query.query_from_sqlite_database import get_metadata_from_sqlite
from ms2query.utils import get_classifier_from_csv_file, column_names_for_output


class ResultsTable:
    default_columns = ["spectrum_ids",
                       "inchikey",
                       "query_precursor_mz",
                       "precursor_mz_difference",
                       "s2v_score",
                       "ms2ds_score",
                       "average_ms2ds_score_for_inchikey14",
                       "nr_of_spectra_with_same_inchikey14",
                       "chemical_neighbourhood_score",
                       "average_tanimoto_score_for_chemical_neighbourhood_score",
                       "nr_of_spectra_for_chemical_neighbourhood_score",
                       "cosine_score",
                       "modified_cosine_score"]

    def __init__(self, preselection_cut_off: int,
                 ms2deepscores: pd.DataFrame,
                 query_spectrum: Spectrum,
                 sqlite_file_name: str,
                 classifier_csv_file_name: Union[str, None] = None,
                 **kwargs):
        # pylint: disable=too-many-arguments

        self.data = pd.DataFrame(columns=self.default_columns, **kwargs)
        self.ms2deepscores = ms2deepscores
        self.preselection_cut_off = preselection_cut_off
        self.query_spectrum = query_spectrum
        self.precursor_mz = query_spectrum.get("precursor_mz")
        self.sqlite_file_name = sqlite_file_name
        self.classifier_csv_file_name = classifier_csv_file_name

    def __eq__(self, other):
        if not isinstance(other, ResultsTable):
            return False

        # Round is used to prevent returning float for float rounding errors
        return other.preselection_cut_off == self.preselection_cut_off and \
            other.precursor_mz == self.precursor_mz and \
            self.data.round(5).equals(other.data.round(5)) and \
            self.ms2deepscores.round(5).equals(other.ms2deepscores.round(5)) and \
            self.query_spectrum.__eq__(other.query_spectrum) and \
            self.sqlite_file_name == other.sqlite_file_name

    def assert_results_table_equal(self, other):
        """Assert if results tables are equal except for the spectrum metadata and sqlite file name"""
        assert isinstance(other, ResultsTable), "Expected ResultsTable"
        assert other.preselection_cut_off == self.preselection_cut_off
        assert other.precursor_mz == self.precursor_mz
        assert self.data.round(5).equals(other.data.round(5))
        assert self.ms2deepscores.round(5).equals(other.ms2deepscores.round(5))
        assert self.query_spectrum.peaks == other.query_spectrum.peaks
        assert self.query_spectrum.losses == other.query_spectrum.losses

    def set_index(self, column_name):
        self.data = self.data.set_index(column_name)

    def add_related_inchikey_scores(self, related_inchikey_scores):
        self.data["chemical_neighbourhood_score"] = \
            [related_inchikey_scores[x][0] for x in self.data["inchikey"]]
        self.data["nr_of_spectra_for_chemical_neighbourhood_score"] = \
            [related_inchikey_scores[x][1] for x in self.data["inchikey"]]
        self.data["average_tanimoto_score_for_chemical_neighbourhood_score"] = \
            [related_inchikey_scores[x][2] for x in self.data["inchikey"]]

    def add_average_ms2ds_scores(self, average_ms2ds_scores):
        self.data["average_ms2ds_score_for_inchikey14"] = \
            [average_ms2ds_scores[x][0] for x in self.data["inchikey"]]
        self.data["nr_of_spectra_with_same_inchikey14"] = \
            [average_ms2ds_scores[x][1] for x in self.data["inchikey"]]

    def add_precursors(self, precursors):
        assert isinstance(precursors, np.ndarray), "Expected np.ndarray as input."
        self.data["query_precursor_mz"] = precursors
        self.data["precursor_mz_difference"] = (np.abs(precursors - self.precursor_mz))

    def preselect_on_ms2deepscore(self):
        selected_spectrum_ids = list(self.ms2deepscores.nlargest(
            self.preselection_cut_off).index)
        self.data["spectrum_ids"] = pd.Series(selected_spectrum_ids)
        self.data["ms2ds_score"] = \
            np.array(self.ms2deepscores.loc[selected_spectrum_ids])

    def add_ms2query_meta_score(self,
                                predictions):
        """Add MS2Query meta score to data and sort on this score

        Args:
        ------
        predictions:
            An iterable containing the ms2query model meta scores
        """
        self.data["ms2query_model_prediction"] = predictions
        self.data.sort_values(by=["ms2query_model_prediction"],
                              ascending=False,
                              inplace=True)

    def add_multiple_structure_scores(self, average_ms2deepscores, closely_related_inchikeys):
        tanimoto_scores_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[],7:[], 8:[],9:[]}
        average_ms2deepscore_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[],7:[], 8:[],9:[]}
        nr_of_spectra_per_inchikey_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[],7:[], 8:[],9:[]}
        for inchikeys_per_spectrum in closely_related_inchikeys:
            for structure_nr, (inchikey, tanimoto_score) in enumerate(inchikeys_per_spectrum):
                average_ms2deepscore, nr_of_spectra_for_inchikey = average_ms2deepscores[inchikey]

                tanimoto_scores_dict[structure_nr].append(tanimoto_score)
                average_ms2deepscore_dict[structure_nr].append(average_ms2deepscore)
                nr_of_spectra_per_inchikey_dict[structure_nr].append(nr_of_spectra_for_inchikey)
        # add to data
        for i in range(10):
            self.data["average_ms2deepscore_" + str(i)] = average_ms2deepscore_dict[i]
            self.data["tanimoto_score_structure_" + str(i)] = tanimoto_scores_dict[i]
            self.data["nr_of_spectra_structure_" + str(i)] = nr_of_spectra_per_inchikey_dict[i]

    def add_instrument_types(self):
        # Add for library spectra
        lib_spectrum_ids = list(self.data.index)
        metadata = get_metadata_from_sqlite(self.sqlite_file_name, lib_spectrum_ids)
        orbitrap_list = []
        iontrap_list = []
        tof_list = []
        quadrupole_list = []
        for lib_spectrum_id in lib_spectrum_ids:
            instrument_type = parse_instrument_type(metadata[lib_spectrum_id]["source_instrument"])
            if instrument_type == "Orbitrap":
                orbitrap_list.append(1)
            else:
                orbitrap_list.append(0)
            if instrument_type == "Ion Trap":
                iontrap_list.append(1)
            else:
                iontrap_list.append(0)
            if instrument_type == "ToF":
                tof_list.append(1)
            else:
                tof_list.append(0)
            if instrument_type == "Quadrupole":
                quadrupole_list.append(1)
            else:
                quadrupole_list.append(0)
        self.data["lib_instrument_orbitrap"] = orbitrap_list
        self.data["lib_instrument_ion_trap"] = iontrap_list
        self.data["lib_instrument_tof"] = tof_list
        self.data["lib_instrument_quadrupole"] = quadrupole_list

        # Add for query spectrum
        query_spec_instrument = self.query_spectrum.get("source_instrument")
        query_instrument_type = parse_instrument_type(query_spec_instrument)
        if query_instrument_type == "Orbitrap":
            self.data["query_instrument_orbitrap"] = [1] * self.preselection_cut_off
        else:
            self.data["query_instrument_orbitrap"] = [0] * self.preselection_cut_off
        if query_instrument_type == "Ion Trap":
            self.data["query_instrument_ion_trap"] = [1] * self.preselection_cut_off
        else:
            self.data["query_instrument_ion_trap"] = [0] * self.preselection_cut_off
        if query_instrument_type == "ToF":
            self.data["query_instrument_tof"] = [1] * self.preselection_cut_off
        else:
            self.data["query_instrument_tof"] = [0] * self.preselection_cut_off
        if query_instrument_type == "Quadrupole":
            self.data["query_instrument_quadrupole"] = [1] * self.preselection_cut_off
        else:
            self.data["query_instrument_quadrupole"] = [0] * self.preselection_cut_off

    def get_training_data(self) -> pd.DataFrame:
        column_list = ["query_precursor_mz",
                       "precursor_mz_difference",
                       "s2v_score",
                       "ms2ds_score",
                       "average_ms2ds_score_for_inchikey14",
                       "nr_of_spectra_with_same_inchikey14",
                       "chemical_neighbourhood_score",
                       "average_tanimoto_score_for_chemical_neighbourhood_score",
                       "nr_of_spectra_for_chemical_neighbourhood_score",
                       "cosine_score",
                       "modified_cosine_score",
                       "lib_instrument_orbitrap",
                       "lib_instrument_ion_trap",
                       "lib_instrument_tof",
                       "lib_instrument_quadrupole",
                       "query_instrument_orbitrap",
                       "query_instrument_ion_trap",
                       "query_instrument_tof",
                       "query_instrument_quadrupole"
                       ]
        for i in range(10):
            column_list.append("average_ms2deepscore_" + str(i))
            column_list.append("tanimoto_score_structure_" + str(i))
            column_list.append("nr_of_spectra_structure_" + str(i))
        return self.data[column_list]

    def export_to_dataframe(
            self,
            nr_of_top_spectra: int,
            minimal_ms2query_score: Union[float, int] = 0.0,
            additional_metadata_columns: List[str] = None,
            additional_ms2query_score_columns: List[str] = None
            ) -> Union[None, pd.DataFrame]:
        """Returns a dataframe with analogs results from results table

        Args:
        ------
        nr_of_top_spectra:
            Number of spectra that should be returned.
            The best spectra are selected based on highest MS2Query meta score
        minimal_ms2query_score:
            Only results with ms2query metascore >= minimal_ms2query_score will be returned.
        additional_metadata_columns:
            Additional columns with query spectrum metadata that should be added. For instance "retention_time".
        additional_ms2query_score_columns:
            Additional columns with scores used for calculating the ms2query metascore
            Options are: "mass_similarity", "s2v_score", "ms2ds_score", "average_ms2ds_score_for_inchikey14",
            "nr_of_spectra_with_same_inchikey14", "chemical_neighbourhood_score",
            "average_tanimoto_score_for_chemical_neighbourhood_score",
            "nr_of_spectra_for_chemical_neighbourhood_score"
        """
        # Select top results
        selected_analogs: pd.DataFrame = \
            self.data.iloc[:nr_of_top_spectra, :].copy()
        selected_analogs.reset_index(inplace=True)

        # Remove analogs that do not have a high enough ms2query score
        selected_analogs = selected_analogs[
            (selected_analogs["ms2query_model_prediction"] > minimal_ms2query_score)]
        nr_of_analogs = len(selected_analogs)
        # Return None if know analogs are selected.
        if selected_analogs.empty:
            return None

        # For each analog the compound name is selected from sqlite
        metadata_dict = get_metadata_from_sqlite(self.sqlite_file_name,
                                                 list(selected_analogs["spectrum_ids"]))
        compound_name_list = [metadata_dict[analog_spectrum_id]["compound_name"]
                              for analog_spectrum_id
                              in list(selected_analogs["spectrum_ids"])]

        # Add inchikey and ms2query model prediction to results df
        # results_df = selected_analogs.loc[:, ["spectrum_ids", "ms2query_model_prediction", "inchikey"]]
        results_df = pd.DataFrame({"spectrum_ids": selected_analogs["spectrum_ids"],
                                   "ms2query_model_prediction": selected_analogs["ms2query_model_prediction"],
                                   "inchikey": selected_analogs["inchikey"],
                                   "precursor_mz_analog": selected_analogs["precursor_mz*0.001"] * 1000,
                                   "precursor_mz_query_spectrum": [self.precursor_mz] * nr_of_analogs,
                                   "analog_compound_name": compound_name_list,
                                   "precursor_mz_difference": abs(selected_analogs["precursor_mz*0.001"] * 1000 -
                                                                 pd.Series([self.precursor_mz] * nr_of_analogs))
                                   })
        if additional_metadata_columns is not None:
            for metadata_name in additional_metadata_columns:
                results_df[metadata_name] = [self.query_spectrum.get(metadata_name)] * nr_of_analogs
        if additional_ms2query_score_columns is not None:
            for score in additional_ms2query_score_columns:
                results_df[score] = selected_analogs[score]

        # Orders the columns in the right way
        results_df = results_df.reindex(
            columns=column_names_for_output(True, False, additional_metadata_columns,
                                            additional_ms2query_score_columns))
        # Add classifiers to dataframe
        if self.classifier_csv_file_name is not None:
            classifiers_df = \
                get_classifier_from_csv_file(self.classifier_csv_file_name,
                                             results_df["inchikey"].unique())
            results_df = pd.merge(results_df,
                                  classifiers_df,
                                  on="inchikey")
        return results_df


def parse_instrument_type(instrument_name):
    # Options: Qtof, Orbitrab, QQQ, ion trap, CID-API, HCD, FTMS, CID, FAB-EBEB, ITFT, APPI-QQ
    if instrument_name == None:
        return None
    found_tof = re.search("(?i)tof", instrument_name)
    if found_tof is not None:
        return "ToF"
    found_ion_trap = re.search("(?i)ion trap", instrument_name)
    if found_ion_trap is not None:
        return "Ion Trap"
    found_ion_trap = re.search("(?i)itft", instrument_name)
    if found_ion_trap is not None:
        return "Ion Trap"
    found_ion_trap = re.search("-IT", instrument_name)
    if found_ion_trap is not None:
        return "Ion Trap"
    found_quadrupole = re.search("(?i)qq", instrument_name)
    if found_quadrupole is not None:
        return "quadrupole"
    found_quadrupole = re.search("(?i)QFT", instrument_name)
    if found_quadrupole is not None:
        return "quadrupole"
    found_orbitrab = re.search("(?i)hybrid ft", instrument_name)
    if found_orbitrab is not None:
        return "Orbitrap"
    found_orbitrab = re.search("(?i)orbitrap", instrument_name)
    if found_orbitrab is not None:
        return "Orbitrap"
    found_orbitrab = re.search("(?i)velos", instrument_name)
    if found_orbitrab is not None:
        return "Orbitrap"
    found_orbitrab = re.search("(?i)lumos", instrument_name)
    if found_orbitrab is not None:
        return "Orbitrap"
    found_orbitrab = re.search("(?i)q-exactive", instrument_name)
    if found_orbitrab is not None:
        return "Orbitrap"