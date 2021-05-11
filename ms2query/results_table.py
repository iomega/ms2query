import numpy as np
import pandas as pd


class ResultsTable(pd.DataFrame):
    default_columns = ["spectrum_ids",
                       "inchikey",
                       "parent_mass*0.001",
                       "mass_similarity",
                       "s2v_score",
                       "ms2ds_score",
                       "average_ms2ds_score_for_inchikey14",
                       "nr_of_spectra_with_same_inchikey14*0.01",
                       "closely_related_inchikey14s_score",
                       "average_tanimoto_for_closely_related_score",
                       "nr_of_spectra_for_closely_related_score*0.01"]

    def __init__(self, preselection_cut_off: int,
                 spectrum_id: str,
                 parent_mass: float,
                 **kwargs):
        super(ResultsTable, self).__init__(self.default_columns, **kwargs)
        self.preselection_cut_off = preselection_cut_off
        self.spectrum_id = spectrum_id
        self.parent_mass = parent_mass

    def add_related_inchikey_scores(self, related_inchikey_scores):
        self["closely_related_inchikey14s_score"] = \
            [related_inchikey_scores[x][0] for x in self["inchikey"]]
        self["nr_of_spectra_for_closely_related_score*0.01"] = \
            [related_inchikey_scores[x][1] / 100 for x in self["inchikey"]]
        self["average_tanimoto_for_closely_related_score"] = \
            [related_inchikey_scores[x][2] for x in self["inchikey"]]

    def add_average_ms2ds_scores(self, average_ms2ds_scores):
        self["average_ms2ds_score_for_inchikey14"] = \
            [average_ms2ds_scores[x][0] for x in self["inchikey"]]
        self["nr_of_spectra_with_same_inchikey14*0.01"] = \
            [average_ms2ds_scores[x][1] / 100 for x in self["inchikey"]]

    def add_parent_masses(self, parent_masses, base_nr_mass_similarity):
        assert isinstance(parent_masses, np.ndarray), "Expected np.ndarray as input."
        self["parent_mass*0.001"] = parent_masses / 1000

        self["mass_similarity"] = base_nr_mass_similarity ** \
            (np.abs(parent_masses - self.parent_mass))