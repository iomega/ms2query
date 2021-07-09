import pickle
import pandas as pd
import numpy as np
from numpy import nan as Nan


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def get_classifier_from_csv_file(csv_file, list_of_inchikeys):
    classifiers_df = pd.read_csv(csv_file, sep="\t")
    columns_to_keep = ["inchi_key", "smiles", "cf_kingdom",
                       "cf_superclass", "cf_class", "cf_subclass",
                       "cf_direct_parent", "npc_class_results",
                       "npc_superclass_results", "npc_pathway_results"]
    list_of_classifiers = []
    for inchikey in list_of_inchikeys:
        classifiers = classifiers_df.loc[
            classifiers_df["inchi_key"].str.startswith(inchikey)]
        if classifiers.empty:
            list_of_classifiers.append(pd.DataFrame(np.array(
                [[inchikey] + [Nan] * (len(columns_to_keep) - 1)])
                , columns=columns_to_keep))
        else:
            classifiers = classifiers[columns_to_keep].iloc[:1]

            list_of_classifiers.append(classifiers)
    if len(list_of_classifiers) == 0:
        results = pd.DataFrame(columns=columns_to_keep)
    else:
        results = pd.concat(list_of_classifiers, axis=0)

    results["inchi_key"] = list_of_inchikeys
    results.rename(columns={"inchi_key": "inchikey"}, inplace=True)
    return results


def add_classifiers_to_df(classifier_csv_file, features_df):
    classifiers_df = \
        get_classifier_from_csv_file(classifier_csv_file,
                                     set(features_df["inchikey"]))
    data = features_df.reset_index()
    data_with_added_classifiers = pd.merge(data,
                                           classifiers_df,
                                           on="inchikey")
    return data_with_added_classifiers
