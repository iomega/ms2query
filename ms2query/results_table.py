import numpy as np
import pandas as pd
from typing import Union, List
from ms2query.query_from_sqlite_database import get_metadata_from_sqlite


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
                 sqlite_file_name,
                 **kwargs):
        self.data = pd.DataFrame(columns=self.default_columns, **kwargs)
        self.preselection_cut_off = preselection_cut_off
        self.ms2deepscores = ms2deepscores
        self.query_spectrum = query_spectrum
        self.parent_mass = query_spectrum.get("parent_mass")
        self.sqlite_file_name = sqlite_file_name

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

    def create_dataframe_with_compound_classification(
            self,
            nr_of_top_spectra,
            classifiers_file_name,
            add_selection: Union[bool, List[str]] = False):
        selected_results = \
            self.data.iloc[:nr_of_top_spectra, :].copy()
        metadata_list = list(get_metadata_from_sqlite(self.sqlite_file_name,
                                       list(selected_results.index)).values())
        compound_names = [metadata["compound_name"] for metadata in metadata_list]
        selected_results["compound_name"] = compound_names
        selected_results = add_classifiers_to_df(classifiers_file_name,
                                                 selected_results)
        if add_selection is not False:
            selected_results = selected_results[add_selection]
        selected_results.sort_values(
            by=["ms2query_model_prediction"], ascending=False, inplace=True)
        selected_results.set_index("spectrum_ids", inplace=True)
        return selected_results


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
    classifiers_df = pd.read_csv(classifier_file_name, sep="\t")
    columns_to_keep = ["inchi_key", "smiles", "cf_kingdom",
                       "cf_superclass", "cf_class", "cf_subclass",
                       "cf_direct_parent", "npc_class_results",
                       "npc_superclass_results", "npc_pathway_results"]
    list_of_classifiers = []
    for inchikey in list_of_inchikeys:
        classifiers = classifiers_df.loc[
            classifiers_df["inchi_key"].str.startswith(inchikey)] # pylint: disable=unsubscriptable-object
        if classifiers.empty:
            list_of_classifiers.append(pd.DataFrame(np.array(
                [[inchikey] + [np.nan] * (len(columns_to_keep) - 1)])
                , columns=columns_to_keep))
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


def add_classifiers_to_df(classifier_csv_file, features_df):
    classifiers_df = \
        get_classifier_from_csv_file(classifier_csv_file,
                                     features_df["inchikey"].unique())
    data = features_df.reset_index()
    data_with_added_classifiers = pd.merge(data,
                                           classifiers_df,
                                           on="inchikey")
    return data_with_added_classifiers
