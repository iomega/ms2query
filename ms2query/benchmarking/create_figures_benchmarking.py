"""
This script is not needed for normally running MS2Query, instead it was used to visualize
test results for benchmarking MS2Query against other tools.
"""
import os
import random
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from ms2query.utils import load_json_file


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


def calculate_recall_vs_tanimoto_scores(selection_criteria_and_tanimoto):
    percentages_found = []
    average_tanimoto_score = []
    # Shuffeling to make sure there is a random order for the matches with the same MS2Query score.
    random.shuffle(selection_criteria_and_tanimoto)
    # remove None values
    not_none_scores = [score for score in selection_criteria_and_tanimoto if score is not None]
    sorted_scores = sorted(not_none_scores, key=lambda tup: tup[0], reverse=True)
    sorted_tanimoto_scores = [scores[1] for scores in sorted_scores]
    for _ in tqdm(range(len(sorted_scores)),
                  desc="Calculating average Tanimoto score"):
        percentages_found.append(len(sorted_tanimoto_scores)/len(selection_criteria_and_tanimoto)*100)
        average_tanimoto_score.append(sum(sorted_tanimoto_scores)/len(sorted_tanimoto_scores))
        sorted_tanimoto_scores.pop()
    return percentages_found, average_tanimoto_score


def calculate_recall_vs_tanimoto_scores_average(selection_criteria_and_tanimoto):
    percentages_found = []
    average_tanimoto_score = []
    # Shuffeling to make sure there is a random order for the matches with the same MS2Query score.
    # random.shuffle(selection_criteria_and_tanimoto)
    # remove None values
    not_none_scores = [score for score in selection_criteria_and_tanimoto if score is not None]
    sorted_scores = sorted(not_none_scores, key=lambda tup: tup[0], reverse=True)
    unique_scores = sorted(set([scores[0] for scores in sorted_scores]))
    # sorted_tanimoto_scores = [scores[1] for scores in sorted_scores]
    for unique_score in tqdm(unique_scores,
                             desc="Calculating average Tanimoto score"):
        tanimoto_scores = [scores[1] for scores in sorted_scores if scores[0] >= unique_score]
        percentages_found.append(len(tanimoto_scores)/len(selection_criteria_and_tanimoto)*100)
        average_tanimoto_score.append(sum(tanimoto_scores)/len(tanimoto_scores))
    return percentages_found, average_tanimoto_score


def avg_tanimoto_vs_percentage_found(percentages_found, average_tanimoto_score,
                                     legend_label,
                                     color_code,
                                     line_style):
    """Plots the average tanimoto vs recall"""
    plt.plot(percentages_found, average_tanimoto_score,
             label=legend_label,
             color=color_code,
             linestyle=line_style)
    plt.xlim(100, 0)
    # plt.ylim(0.4, 1)
    plt.xlabel("Recall (%)")
    plt.ylabel("Average Tanimoto score")
    plt.suptitle("Analogues test set")
    plt.legend(loc="lower right",
               title="Select on:")


def load_results_from_folder(test_results_folder: str):
    dict_with_results = {}
    dict_with_results["MS2Query"] = load_json_file(os.path.join(test_results_folder, "ms2query_test_results.json"))
    dict_with_results["MS2Deepscore"] = load_json_file(os.path.join(test_results_folder, "ms2deepscore_test_results_all.json"))
    dict_with_results["MS2Deepscore 100 Da"] = load_json_file(os.path.join(test_results_folder, "ms2deepscore_test_results_100_Da.json"))
    dict_with_results["Cosine score 100 Da"] = load_json_file(os.path.join(test_results_folder, "cosine_score_100_da_test_results.json"))
    dict_with_results["Modified cosine score 100 Da"] = load_json_file(os.path.join(test_results_folder, "modified_cosine_score_100_Da_test_results.json"))
    dict_with_results["Optimal"] = load_json_file(os.path.join(test_results_folder, "optimal_results.json"))
    dict_with_results["Random"] = load_json_file(os.path.join(test_results_folder, "random_results.json"))
    return dict_with_results


def plot_results(dict_with_results):
    colours = {"MS2Query": ('#3C5289', "-"),
               "MS2Deepscore": ('#49C16D', "-"),
               "MS2Deepscore 100 Da": ('#49C16D', "-"),
               "Cosine score 100 Da": ('#F5E21D', "-"),
               "Modified cosine score 100 Da": ('#F5E21D', "-"),
               "Optimal": ('#000000', "--"),
               "Random": ('#808080', "--")}
    for results_name, scores in dict_with_results.items():
        percentages_found, average_tanimoto_score = calculate_recall_vs_tanimoto_scores(
            scores)
        avg_tanimoto_vs_percentage_found(percentages_found, average_tanimoto_score, results_name, colours[results_name][0], colours[results_name][1])
    plt.show()


def combine_all_test_results():
    base_directory = "../../data/libraries_and_models/gnps_01_11_2022/20_fold_splits/"
    results_dict = {"MS2Query": [],
                    "MS2Deepscore": [],
                    "MS2Deepscore 100 Da": [],
                    "Cosine score 100 Da": [],
                    "Modified cosine score 100 Da": [],
                    "Optimal": [],
                    "Random": []}
    for i in range(5):
        test_results_directory = os.path.join(base_directory, f"test_split_{i}", "test_results")
        try:
            results = load_results_from_folder(test_results_directory)
            for key in results_dict:
                results_dict[key] += results[key]
        except:
            print(f"not all test results were generated for : {test_results_directory}")
    return results_dict


if __name__ == "__main__":
    # test_results_folder = "../../data/test_dir/train_and_test_library/test_results/"
    # test_results_folder = "../../data/libraries_and_models/gnps_01_11_2022/20_fold_splits/test_split_0/test_results/"
    dict_with_results = combine_all_test_results()
    plot_results(dict_with_results)
