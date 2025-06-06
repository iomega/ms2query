from typing import Tuple, Union
import numpy as np
import pandas as pd
from matchms.Spectrum import Spectrum
from ms2query.query_from_sqlite_database import SqliteLibrary
from ms2query.utils import column_names_for_output


class ResultsTable:
    default_columns = ["spectrum_ids",
                       "inchikey",
                       "precursor_mz_library_spectrum",
                       "precursor_mz_difference",
                       "s2v_score",
                       "ms2ds_score",
                       "average_ms2deepscore_multiple_library_structures",
                       "average_tanimoto_score_library_structures"]

    def __init__(self, preselection_cut_off: int,
                 ms2deepscores: pd.Series,
                 query_spectrum: Spectrum,
                 sqlite_library: SqliteLibrary,
                 **kwargs):

        self.data = pd.DataFrame(columns=self.default_columns, **kwargs)
        self.ms2deepscores = ms2deepscores
        self.preselection_cut_off = preselection_cut_off
        self.query_spectrum = query_spectrum
        self.precursor_mz = query_spectrum.get("precursor_mz")
        self.sqlite_library = sqlite_library

    def __eq__(self, other):
        if not isinstance(other, ResultsTable):
            return False

        # Round is used to prevent returning float for float rounding errors
        # We cannot check the sqlite file location, since this will have a different path on a virtual machine.
        return other.preselection_cut_off == self.preselection_cut_off and \
            other.precursor_mz == self.precursor_mz and \
            self.data.round(5).equals(other.data.round(5)) and \
            self.ms2deepscores.round(5).equals(other.ms2deepscores.round(5)) and \
            self.query_spectrum.__eq__(other.query_spectrum)

    def assert_results_table_equal(self, other):
        """Assert if results tables are equal except for the spectrum metadata and sqlite file name"""
        assert isinstance(other, ResultsTable), "Expected ResultsTable"
        assert other.preselection_cut_off == self.preselection_cut_off
        assert other.precursor_mz == self.precursor_mz
        pd.testing.assert_frame_equal(self.data, other.data, check_dtype=False)
        assert self.ms2deepscores.round(5).equals(other.ms2deepscores.round(5)), f"ms2deepscores are not equal {self.ms2deepscores} != {other.ms2deepscores}"
        assert self.query_spectrum.peaks == other.query_spectrum.peaks
        assert self.query_spectrum.losses == other.query_spectrum.losses
        # We cannot check the sqlite file location, since this will have a different path on a virtual machine.

    def set_index(self, column_name):
        self.data = self.data.set_index(column_name)

    def add_related_inchikey_scores(self, related_inchikey_scores):

        self.data["average_ms2deepscore_multiple_library_structures"] = \
            [related_inchikey_scores[x][0] for x in self.data["inchikey"]]
        self.data["average_tanimoto_score_library_structures"] = \
            [related_inchikey_scores[x][1] for x in self.data["inchikey"]]

    def add_precursors(self, precursors):
        assert isinstance(precursors, np.ndarray), "Expected np.ndarray as input."
        self.data["precursor_mz_library_spectrum"] = precursors
        self.data["precursor_mz_difference"] = np.abs(precursors - self.precursor_mz)

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

    def get_training_data(self) -> pd.DataFrame:
        return self.data[["precursor_mz_library_spectrum",
                          "precursor_mz_difference",
                          "s2v_score",
                          "average_ms2deepscore_multiple_library_structures",
                          "average_tanimoto_score_library_structures"]]

    def export_to_dataframe(
            self,
            nr_of_top_spectra: int,
            minimal_ms2query_score: Union[float, int] = 0.0,
            additional_metadata_columns: Tuple[str, ...] = None,
            additional_ms2query_score_columns: Tuple[str, ...] = None
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
            Options are: "s2v_score", "ms2ds_score", "average_ms2deepscore_multiple_library_structures",
            "average_tanimoto_score_library_structures"
        """
        # Select top results
        selected_analogs: pd.DataFrame = \
            self.data.iloc[:nr_of_top_spectra, :].copy()
        selected_analogs.reset_index(inplace=True)

        # Remove analogs that do not have a high enough ms2query score
        selected_analogs = selected_analogs[
            (selected_analogs["ms2query_model_prediction"] > minimal_ms2query_score)]
        nr_of_analogs = len(selected_analogs)
        # Return None if no analogs are selected.
        if selected_analogs.empty:
            return None

        # For each analog the compound name is selected from sqlite
        metadata_dict = self.sqlite_library.get_metadata_from_sqlite(list(selected_analogs["spectrum_ids"]))

        compound_name_list = []
        for metadata in metadata_dict.values():
            if "compound_name" in metadata.keys():
                compound_name_list.append(metadata["compound_name"])
            else:
                compound_name_list.append(None)

        smiles_list = []
        for metadata in metadata_dict.values():
            if "smiles" in metadata.keys():
                smiles_list.append(metadata["smiles"])
            else:
                smiles_list.append(None)

        # Add inchikey and ms2query model prediction to results df
        # results_df = selected_analogs.loc[:, ["spectrum_ids", "ms2query_model_prediction", "inchikey"]]
        results_df = pd.DataFrame({"query_spectrum_nr": self.query_spectrum.get("spectrum_nr"),
                                   # "spectrum_ids": selected_analogs["spectrum_ids"],
                                   "ms2query_model_prediction": selected_analogs["ms2query_model_prediction"],
                                   "inchikey": selected_analogs["inchikey"],
                                   "precursor_mz_analog": selected_analogs["precursor_mz_library_spectrum"],
                                   "precursor_mz_query_spectrum": [self.precursor_mz] * nr_of_analogs,
                                   "smiles": smiles_list,
                                   "analog_compound_name": compound_name_list,
                                   "precursor_mz_difference": selected_analogs["precursor_mz_difference"],
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
        if self.sqlite_library.contains_class_annotation():
            classifiers_df = self.sqlite_library.get_classes_inchikeys(results_df["inchikey"].unique())
            results_df = pd.merge(results_df,
                                  classifiers_df,
                                  on="inchikey")
        return results_df
