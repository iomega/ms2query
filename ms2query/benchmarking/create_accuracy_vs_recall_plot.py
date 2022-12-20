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

from ms2query.utils import load_json_file, load_pickled_file, save_pickled_file


def plot_all_with_standard_deviation(means_and_standars_deviation,
                                     save_figure_file_name=None,
                                     ):
    colours = {
        "MS2Query": ('#3C5289', "-"),
        "MS2Deepscore 100 Da": ('#49C16D', "-"),
        "Cosine score 100 Da": ('#FFA500', "-"),
        "Modified cosine score 100 Da": ('#F5E21D', "-"),
        # "MS2Deepscore 0.25 Da": ('#49C16D', "-"),
        # "Cosine score 0.25 Da": ('#FFA500', "-"),
        # "Modified cosine score 0.25 Da": ('#F5E21D', "-"),
        "Optimal": ('#000000', "--"),
        "Random": ('#808080', "--")
    }

    for test_type, colour_and_line_type in colours.items():
        colour = colour_and_line_type[0]
        line_type = colour_and_line_type[1]
        binned_percentages, means, standard_deviations = means_and_standars_deviation[test_type]
        plt.plot(binned_percentages, means,
                 label=test_type,
                 color=colour,
                 linestyle=line_type)
        plt.fill_between(binned_percentages, means - standard_deviations, means + standard_deviations, alpha=0.3,
                         color=colour,
                         )

    plt.xlim(100, 0)
    plt.ylim(0, 1.05)
    plt.xlabel("Recall (%)")
    plt.ylabel("Average Tanimoto score")
    plt.suptitle("Analogues test set")
    plt.legend(loc="lower left")
    if save_figure_file_name is None:
        plt.show()
    else:
        assert not os.path.isfile(save_figure_file_name)
        plt.savefig(save_figure_file_name, format="svg")


def plot_exact_matches_results_with_standard_deviation(means_and_standars_deviation,
                                                       save_figure_file_name=None):
    colours = {
        "MS2Query": ('#3C5289', "-"),
        "MS2Deepscore 0.25 Da": ('#49C16D', "-"),
        "Cosine score 0.25 Da": ('#FFA500', "-"),
        "Modified cosine score 0.25 Da": ('#F5E21D', "-"),
        "Optimal": ('#000000', "--"),
        "Random": ('#808080', "--")
    }

    for test_type, colour_and_line_type in colours.items():
        colour = colour_and_line_type[0]
        line_type = colour_and_line_type[1]
        binned_percentages, means, standard_deviations = means_and_standars_deviation[test_type]
        plt.plot(binned_percentages, means,
                 label=test_type,
                 color=colour,
                 linestyle=line_type)
        plt.fill_between(binned_percentages, means - standard_deviations, means + standard_deviations, alpha=0.3,
                         color=colour,
                         )

    plt.xlim(100, 0)
    # plt.ylim(0, 100)
    plt.xlabel("Recall (%)")
    plt.ylabel("True matches (%)")
    plt.suptitle("Exact matches test set")
    plt.legend(loc="lower left")
    if save_figure_file_name is None:
        plt.show()
    else:
        assert not os.path.isfile(save_figure_file_name)
        plt.savefig(save_figure_file_name, format="svg")


def calculate_all_means_and_standard_deviation(dict_with_results,
                                               exact_matches: bool = False):
    means_and_standard_deviation = {}
    for results_name, scores in dict_with_results.items():
        binned_percentages, means, standard_deviations = calculate_means_and_standard_deviation(scores,
                                                                                                exact_matches=exact_matches)
        means_and_standard_deviation[results_name] = (binned_percentages, means, standard_deviations)
    return means_and_standard_deviation


def load_all_test_results(nr_of_test_results, base_directory, exact_match=False):
    results_dict = {}
    for i in range(nr_of_test_results):
        test_results_directory = os.path.join(base_directory, f"test_split_{i}", "test_results")
        try:
            results = load_results_from_folder(test_results_directory,
                                               exact_matches=exact_match)
            for key in results:  # pylint: disable=consider-using-dict-items
                if key in results_dict:
                    results_dict[key].append(results[key])
                else:
                    results_dict[key] = [results[key]]
        except IOError:
            print(f"not all test results were generated for : {test_results_directory}")
    return results_dict


def load_results_from_folder(test_results_folder: str,
                             exact_matches = False) -> Dict[str, List[Tuple[float, float, bool]]]:
    dict_with_results = {}
    if not exact_matches:
        dict_with_results["Cosine score 100 Da"] = load_json_file(os.path.join(test_results_folder, "cosine_score_100_da_test_results.json"))
        dict_with_results["MS2Deepscore 100 Da"] = load_json_file(os.path.join(test_results_folder, "ms2deepscore_test_results_100_Da.json"))
        dict_with_results["Modified cosine score 100 Da"] = load_json_file(os.path.join(test_results_folder, "modified_cosine_score_100_Da_test_results.json"))
    else:
        dict_with_results["MS2Deepscore 0.25 Da"] = load_json_file(
            os.path.join(test_results_folder, "ms2deepscore_test_results_0_25_Da.json"))
        dict_with_results["Cosine score 0.25 Da"] = load_json_file(
            os.path.join(test_results_folder, "cosine_score_0_25_da_test_results.json"))
        dict_with_results["Modified cosine score 0.25 Da"] = load_json_file(
            os.path.join(test_results_folder, "modified_cosine_score_0_25_Da_test_results.json"))

    dict_with_results["MS2Query"] = load_json_file(os.path.join(test_results_folder, "ms2query_test_results.json"))
    dict_with_results["MS2Deepscore"] = load_json_file(
        os.path.join(test_results_folder, "ms2deepscore_test_results_all.json"))
    dict_with_results["Optimal"] = load_json_file(os.path.join(test_results_folder, "optimal_results.json"))
    dict_with_results["Random"] = load_json_file(os.path.join(test_results_folder, "random_results.json"))
    return dict_with_results


def calculate_recall_vs_exact_matches(selection_criteria_and_tanimoto):
    percentages_found = []
    percentage_exact_match = []
    # Shuffeling to make sure there is a random order for the matches with the same MS2Query score.
    random.shuffle(selection_criteria_and_tanimoto)
    # remove None values
    not_none_scores = [score for score in selection_criteria_and_tanimoto if score is not None]
    sorted_scores = sorted(not_none_scores, key=lambda tup: tup[0], reverse=True)
    sorted_exact_match_bool = [scores[2] for scores in sorted_scores]
    for _ in range(len(sorted_scores)):
        percentages_found.append(len(sorted_exact_match_bool)/len(selection_criteria_and_tanimoto)*100)
        percentage_exact_match.append(sum(sorted_exact_match_bool)/len(sorted_exact_match_bool)*100)
        sorted_exact_match_bool.pop()
    return percentages_found, percentage_exact_match


def calculate_means_and_standard_deviation(k_fold_results: List[List[Tuple[float, float, bool]]],
                                           step_size=0.1,
                                           exact_matches: bool = False):
    accuracies = []
    binned_percentages = np.arange(0 + step_size/2, 100 + step_size/2, step_size)
    for results in tqdm(k_fold_results,
                        desc="Calculating recall, mean and standard deviation"):
        if exact_matches:
            percentages_found, average_tanimoto = calculate_recall_vs_exact_matches(results)
        else:
            percentages_found, average_tanimoto = calculate_recall_vs_tanimoto_scores(results)
        binned_accuracies = bin_percentages(percentages_found, average_tanimoto, step_size)
        accuracies.append(binned_accuracies)
    accuracies_matrix = np.vstack(accuracies)
    means = accuracies_matrix.mean(axis=0)
    standard_deviations = accuracies_matrix.std(axis=0, ddof=1)
    return binned_percentages, means, standard_deviations


def bin_percentages(percentages, accuracies, step_size):
    assert len(percentages) == len(accuracies)
    binned_accuracies = []
    binned_percentages = list(np.arange(0, 100, step_size))
    for lower_end_of_bin in binned_percentages:
        upper_end_of_bin = lower_end_of_bin + step_size
        accuracies_in_bin = []
        for i, percentage_found in enumerate(percentages):
            if lower_end_of_bin < percentage_found <= upper_end_of_bin:
                accuracies_in_bin.append(accuracies[i])

        if len(accuracies_in_bin) != 0:
            average = sum(accuracies_in_bin)/len(accuracies_in_bin)
        binned_accuracies.append(average)
    return binned_accuracies


def calculate_recall_vs_tanimoto_scores(selection_criteria_and_tanimoto):
    percentages_found = []
    average_tanimoto_score = []
    # Shuffeling to make sure there is a random order for the matches with the same MS2Query score.
    random.shuffle(selection_criteria_and_tanimoto)
    # remove None values
    not_none_scores = [score for score in selection_criteria_and_tanimoto if score is not None]
    sorted_scores = sorted(not_none_scores, key=lambda tup: tup[0], reverse=True)
    sorted_tanimoto_scores = [scores[1] for scores in sorted_scores]
    for _ in range(len(sorted_scores)):
        percentages_found.append(len(sorted_tanimoto_scores)/len(selection_criteria_and_tanimoto)*100)
        average_tanimoto_score.append(sum(sorted_tanimoto_scores)/len(sorted_tanimoto_scores))
        sorted_tanimoto_scores.pop()
    return percentages_found, average_tanimoto_score


def create_plot(exact_matches,
                positive,
                recalculate_means,
                save_figure,
                base_folder):
    if positive:
        if exact_matches:
            test_results_folder = os.path.join(base_folder, "positive_mode/exact_matches_test_sets_splits")
        else:
            test_results_folder = os.path.join(base_folder, "positive_mode/analogue_test_sets_splits")
    else:
        if not exact_matches:
            test_results_folder = os.path.join(base_folder, "negative_mode/analogue_test_sets_splits")
        else:
            test_results_folder = os.path.join(base_folder, "negative_mode/exact_matches_test_sets_splits")

    if exact_matches:
        extra_file_name = "_exact_matches"
    else:
        extra_file_name = ""

    if recalculate_means:
        dict_with_results = load_all_test_results(20, test_results_folder, exact_match=exact_matches)
        means_and_standard_deviation = calculate_all_means_and_standard_deviation(dict_with_results, exact_matches=exact_matches)
        save_pickled_file(means_and_standard_deviation,
                          os.path.join(test_results_folder, f"means_and_standard_deviations_20_fold{extra_file_name}.pickle"))
    else:
        means_and_standard_deviation = load_pickled_file(os.path.join(test_results_folder, f"means_and_standard_deviations_20_fold{extra_file_name}.pickle"))

    if exact_matches:
        optimal_results = means_and_standard_deviation["Optimal"]
        new_optimal_results = (optimal_results[0], np.full(1000, 100), np.full(1000, 0))
        means_and_standard_deviation["Optimal"] = new_optimal_results
    if not exact_matches:
        if save_figure:
            plot_all_with_standard_deviation(means_and_standard_deviation,
                                             save_figure_file_name=os.path.join(test_results_folder,
                                                                                "recall_vs_accuracy_final.svg")
                                             )
        else:
            plot_all_with_standard_deviation(means_and_standard_deviation)
    else:
        if save_figure:
            plot_exact_matches_results_with_standard_deviation(means_and_standard_deviation,
                                                               save_figure_file_name=os.path.join(test_results_folder,
                                                                                                  "recall_vs_accuracy_final.svg")
                                             )
        else:
            plot_exact_matches_results_with_standard_deviation(means_and_standard_deviation)


if __name__ == "__main__":
    # The folder where the benchmarking results are stored.
    # These can be downloaded from https://zenodo.org/record/7427094
    base_folder = "../../data/libraries_and_models/gnps_01_11_2022"

    create_plot(exact_matches=True,
                positive=False,
                recalculate_means=True,
                save_figure=False,
                base_folder=base_folder
                )
