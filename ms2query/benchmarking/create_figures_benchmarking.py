import os
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
from matplotlib import pyplot as plt
from ms2query.utils import load_pickled_file


def select_threshold_for_recall(predictions: List[Tuple[float, float, bool]],
                                recall, nr_of_spectra):
    threshold = 0
    step = 0.0001
    stop = False
    found_recall = 0
    while not stop:
        found_recall = len(select_scores_within_threshold(predictions, threshold)) / nr_of_spectra
        if found_recall > recall:
            threshold += step
        else:
            stop = True
    return threshold, found_recall


def select_scores_within_threshold(scores: List[Tuple[float, float, bool]],
                                   threshold):
    return [scores[1] for scores in scores if scores[0] > threshold]


def compare_tanimoto_score_distribution(predictions_and_scores: Dict[str, List[Tuple[float, float, bool]]],
                                        recall,
                                        nr_of_spectra):
    weight_to_convert_to_percentage = 100 / nr_of_spectra

    all_selected_scores = []
    all_labels = []
    all_weights = []
    for method_name, prediction_and_score in predictions_and_scores.items():
        threshold, recall = select_threshold_for_recall(prediction_and_score, recall, nr_of_spectra)
        selected_scores = select_scores_within_threshold(prediction_and_score, threshold)
        weights = [weight_to_convert_to_percentage] * len(selected_scores)
        print(f"{method_name} Threshold: {threshold:.4f} Recall: {recall:.3f}")
        all_selected_scores.append(selected_scores)
        all_labels.append(method_name)
        all_weights.append(weights)

    plt.hist(all_selected_scores,
             bins=np.linspace(0, 1, 11),
             label=all_labels,
             weights=all_weights)

    plt.legend(loc="upper center", title="Select on:")
    plt.xlabel("tanimoto_score")
    plt.ylabel("Percentage of matches (%)")
    # plt.ylim(0, 10)
    plt.show()


def avg_tanimoto_vs_percentage_found(selection_criteria_and_tanimoto,
                                     legend_label,
                                     color_code,
                                     line_style):
    """Plots the average tanimoto vs recall"""
    percentages_found = []
    average_tanimoto_score = []
    sorted_scores = sorted(selection_criteria_and_tanimoto, key=lambda tup: tup[0])
    for i in range(len(sorted_scores)):
        selected_scores = [scores[1] for scores in sorted_scores[i:]]
        percentages_found.append(len(selected_scores)/len(selection_criteria_and_tanimoto)*100)
        average_tanimoto_score.append(sum(selected_scores)/len(selected_scores))
    plt.plot(percentages_found, average_tanimoto_score,
             label=legend_label,
             color=color_code,
             linestyle=line_style)
    plt.xlim(100, 10)
#     plt.ylim(0.4, 1)
    plt.xlabel("Recall (%)")
    plt.ylabel("Average Tanimoto score")
    plt.suptitle("Analogues test set")
    plt.legend(loc="lower right",
               title="Select on:")


def add_colour_and_linetype_labels(test_results):
    pass


if __name__ == "__main__":
    from ms2query.utils import load_json_file
    test_results = load_json_file("../../data/test_dir/test_generate_test_results/test_results.json")
    ms2query_test_results = test_results["ms2query_results"]
    cosine_test_results = test_results["cosine_results"]
    modified_cosine_test_results = test_results["modified_cosine_results"]
    ms2deepscore_test_results = test_results["ms2ds_results"]
    avg_tanimoto_vs_percentage_found(ms2query_test_results, "MS2Query", '#3C5289', "-")
    avg_tanimoto_vs_percentage_found(ms2deepscore_test_results, "MS2Deepscore", '#49C16D', "-")
    avg_tanimoto_vs_percentage_found(modified_cosine_test_results, "Modified Cosine", '#F5E21D', "-")
    plt.show()
    # avg_tanimoto_vs_percentage_found(optimal_results, "Optimal", '#000000', "--")
    # avg_tanimoto_vs_percentage_found(random_results, "Random", '#808080', "--")
    # compare_tanimoto_score_distribution(test_results,
    #                                     0.7, len(list(test_results.values())[0]))