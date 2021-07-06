import numpy as np
import pandas as pd


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
                 query_spectrum,
                 **kwargs):
        self.data = pd.DataFrame(columns=self.default_columns, **kwargs)
        self.preselection_cut_off = preselection_cut_off
        self.ms2deepscores = ms2deepscores
        self.query_spectrum = query_spectrum
        self.parent_mass = query_spectrum.get("parent_mass")

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
        return self.data.drop("inchikey", axis=1)
