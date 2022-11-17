from typing import List, Tuple, Dict
import numpy as np
from matplotlib import pyplot as plt


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