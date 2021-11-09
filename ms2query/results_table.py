import numpy as np
import pandas as pd
from typing import Union
from matchms.Spectrum import Spectrum
from ms2query.query_from_sqlite_database import get_metadata_from_sqlite
from ms2query.utils import get_classifier_from_csv_file


class ResultsTable:
    default_columns = ["spectrum_ids",
                       "inchikey",
                       "parent_mass*0.001",
                       "mass_similarity",
                       "s2v_score",
                       "ms2ds_score",
                       "average_ms2ds_score_for_inchikey14",
                       "nr_of_spectra_with_same_inchikey14*0.01",
                       "chemical_neighbourhood_score",
                       "average_tanimoto_score_for_chemical_neighbourhood_score",
                       "nr_of_spectra_for_chemical_neighbourhood_score*0.01"]

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
        self.parent_mass = query_spectrum.get("parent_mass")
        self.sqlite_file_name = sqlite_file_name
        self.classifier_csv_file_name = classifier_csv_file_name

    def __eq__(self, other):
        if not isinstance(other, ResultsTable):
            return False

        # Round is used to prevent returning float for float rounding errors
        return other.preselection_cut_off == self.preselection_cut_off and \
            other.parent_mass == self.parent_mass and \
            self.data.round(5).equals(other.data.round(5)) and \
            self.ms2deepscores.round(5).equals(other.ms2deepscores.round(5)) and \
            self.query_spectrum.__eq__(other.query_spectrum) and \
            self.sqlite_file_name == other.sqlite_file_name

    def assert_results_table_equal(self, other):
        """Assert if results tables are equal except for the spectrum metadata and sqlite file name"""
        assert isinstance(other, ResultsTable), "Expected ResultsTable"
        assert other.preselection_cut_off == self.preselection_cut_off
        assert other.parent_mass == self.parent_mass
        assert self.data.round(5).equals(other.data.round(5))
        assert self.ms2deepscores.round(5).equals(other.ms2deepscores.round(5))
        assert self.query_spectrum.peaks == other.query_spectrum.peaks
        assert self.query_spectrum.losses == other.query_spectrum.losses

    def set_index(self, column_name):
        self.data = self.data.set_index(column_name)

    def add_related_inchikey_scores(self, related_inchikey_scores):
        self.data["chemical_neighbourhood_score"] = \
            [related_inchikey_scores[x][0] for x in self.data["inchikey"]]
        self.data["nr_of_spectra_for_chemical_neighbourhood_score*0.01"] = \
            [related_inchikey_scores[x][1] / 100 for x in self.data["inchikey"]]
        self.data["average_tanimoto_score_for_chemical_neighbourhood_score"] = \
            [related_inchikey_scores[x][2] for x in self.data["inchikey"]]

    def add_average_ms2ds_scores(self, average_ms2ds_scores):
        self.data["average_ms2ds_score_for_inchikey14"] = \
            [average_ms2ds_scores[x][0] for x in self.data["inchikey"]]
        self.data["nr_of_spectra_with_same_inchikey14*0.01"] = \
            [average_ms2ds_scores[x][1] / 100 for x in self.data["inchikey"]]

    def add_parent_masses(self, parent_masses, base_nr_mass_similarity):
        assert isinstance(parent_masses, np.ndarray), "Expected np.ndarray as input."
        self.data["parent_mass*0.001"] = parent_masses / 1000

        self.data["mass_similarity"] = base_nr_mass_similarity ** \
            (np.abs(parent_masses - self.parent_mass))

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
        return self.data[["parent_mass*0.001",
                          "mass_similarity",
                          "s2v_score",
                          "ms2ds_score",
                          "average_ms2ds_score_for_inchikey14",
                          "nr_of_spectra_with_same_inchikey14*0.01",
                          "chemical_neighbourhood_score",
                          "average_tanimoto_score_for_chemical_neighbourhood_score",
                          "nr_of_spectra_for_chemical_neighbourhood_score*0.01"]]

    def export_to_dataframe(
            self,
            nr_of_top_spectra: int,
            minimal_ms2query_score: Union[float, int] = 0.0
            ) -> Union[None, pd.DataFrame]:
        """Returns a dataframe with analogs results from results table

        Args:
        ------
        nr_of_top_spectra:
            Number of spectra that should be returned.
            The best spectra are selected based on highest MS2Query meta score
        minimal_ms2query_score:
            Only results with ms2query metascore >= minimal_ms2query_score will be returned.
        """
        # Select top results
        selected_analogs: pd.DataFrame = \
            self.data.iloc[:nr_of_top_spectra, :].copy()

        # Remove analogs that do not have a high enough ms2query score
        selected_analogs = selected_analogs[
            (selected_analogs["ms2query_model_prediction"] > minimal_ms2query_score)]
        # Return None if know analogs are selected.
        if selected_analogs.empty:
            return None
        # Add inchikey and ms2query model prediction to results df
        results_df = selected_analogs.loc[:, ["ms2query_model_prediction",
                                              "inchikey"]]

        # Add the parent masses of the analogs
        results_df.insert(1, "parent_mass_analog", selected_analogs["parent_mass*0.001"] * 1000)
        # Add the parent mass of the query spectrum
        results_df.insert(0, "parent_mass_query_spectrum", [self.parent_mass] * nr_of_top_spectra)
        # For each analog the compound name is selected from sqlite
        metadata_dict = get_metadata_from_sqlite(self.sqlite_file_name,
                                                 list(results_df.index))
        compound_name_list = [metadata_dict[analog_spectrum_id]["compound_name"]
                              for analog_spectrum_id
                              in list(results_df.index)]
        results_df["analog_compound_name"] = compound_name_list
        # Removes index and reorders columns so spectrum_id is not the first column
        results_df.reset_index(inplace=True)
        results_df = results_df.iloc[:, [1, 2, 3, 4, 0, 5]]
        # Add classifiers to dataframe
        if self.classifier_csv_file_name is not None:
            classifiers_df = \
                get_classifier_from_csv_file(self.classifier_csv_file_name,
                                             results_df["inchikey"].unique())
            # data = results_df.reset_index()
            results_df = pd.merge(results_df,
                                  classifiers_df,
                                  on="inchikey")

        return results_df
