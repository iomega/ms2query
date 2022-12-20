import os
from typing import Dict, List, Tuple
from matchms import Spectrum
from create_accuracy_vs_recall_plot import load_results_from_folder, \
    calculate_means_and_standard_deviation
from ms2query.utils import load_matchms_spectrum_objects_from_file, load_pickled_file
from matplotlib import pyplot as plt


def split_results_mass(list_of_test_spectra: List[List[Spectrum]],
                       list_of_test_results: List[List[Tuple[float, float, bool]]],
                       bin_borders) -> Dict[str, List[List[Tuple[float, float, bool]]]]:
    # pylint: disable=too-many-locals
    assert len(list_of_test_spectra) == len(list_of_test_results)

    results_all_bins = {}
    for i in range(len(bin_borders) - 1):
        results_all_bins[f"{bin_borders[i]}-{bin_borders[i + 1]}"] = [[] for i in range(len(list_of_test_results))]

    for test_spec_id, test_spectra in enumerate(list_of_test_spectra):
        test_results = list_of_test_results[test_spec_id]
        assert len(test_spectra) == len(test_results)
        bins = {}
        for i in range(len(bin_borders) - 1):
            bins[f"{bin_borders[i]}-{bin_borders[i + 1]}"] = []
        for spec_id, test_spectrum in enumerate(test_spectra):
            mass = test_spectrum.get("precursor_mz")
            added = False
            for j in range(len(bin_borders)-1):
                lower_border = bin_borders[j]
                higher_border = bin_borders[j+1]
                if lower_border <= mass < higher_border:
                    results_all_bins[f"{lower_border}-{higher_border}"][test_spec_id].append(test_results[spec_id])
                    if added:
                        print("was already added")
                    added = True
            if not added:
                print(f"The mass {mass} was not added")
    return results_all_bins


def split_results_mass_all_results(dict_with_results: Dict[str, List[List[Tuple[float, float, bool]]]],
                                   test_spectra: List[List[Spectrum]],
                                   exact_matches: bool = False
                                   ):
    means_and_standard_deviation = {}
    bin_borders = [0, 300, 600, 5000]
    for results_name, scores in dict_with_results.items():
        scores_per_mass = split_results_mass(test_spectra, scores, bin_borders=bin_borders)

        per_mass_means = {}
        for mass_bin, mass_specific_scores in scores_per_mass.items():
            binned_percentages, means, standard_deviations = calculate_means_and_standard_deviation(mass_specific_scores,
                                                                                                    exact_matches=exact_matches)
            per_mass_means[mass_bin] = (binned_percentages, means, standard_deviations)
        means_and_standard_deviation[results_name] = per_mass_means
    return means_and_standard_deviation


def load_all_test_results_and_test_spectra(nr_of_test_results, base_directory
                                           ) -> Tuple[Dict[str, List[List[Tuple[float, float, bool]]]], List[List[Spectrum]]]:
    results_dict = {"MS2Query": [],
                    "MS2Deepscore": [],
                    "MS2Deepscore 100 Da": [],
                    "Cosine score 100 Da": [],
                    "Modified cosine score 100 Da": [],
                    "Optimal": [],
                    "Random": []}
    all_test_spectra = []
    for i in range(nr_of_test_results):
        test_results_directory = os.path.join(base_directory, f"test_split_{i}", "test_results")
        try:
            results = load_results_from_folder(test_results_directory)
            for key in results_dict:  # pylint: disable=consider-using-dict-items
                results_dict[key].append(results[key])
            test_spectra = load_matchms_spectrum_objects_from_file(
                os.path.join(base_directory, f"test_split_{i}", "test_spectra.pickle"))
            all_test_spectra.append(test_spectra)
        except IOError:
            print(f"not all test results were generated for : {test_results_directory}")
    return results_dict, all_test_spectra


def plot_all_with_standard_deviation_mass(all_means_and_standars_deviation,
                                          type_to_plot,
                                          save_figure_file_name=None):
    means_and_standars_deviation = all_means_and_standars_deviation[type_to_plot]
    for test_type in means_and_standars_deviation:
        binned_percentages, means, standard_deviations = means_and_standars_deviation[test_type]
        plt.plot(binned_percentages, means,
                 label=test_type)
        plt.fill_between(binned_percentages, means - standard_deviations, means + standard_deviations, alpha=0.3)
    plt.xlim(100, 0)
    plt.ylim(0, 1.05)
    plt.xlabel("Recall (%)")
    plt.ylabel("Average Tanimoto score")
    plt.suptitle(f"{type_to_plot}")
    plt.legend(loc="lower left", title="Mass bin (Da)")
    if save_figure_file_name is None:
        plt.show()
    else:
        assert not os.path.isfile(save_figure_file_name)
        plt.savefig(save_figure_file_name, format="svg")
        plt.show()


if __name__ == "__main__":
    test_results_folder = "../../data/libraries_and_models/gnps_01_11_2022/20_fold_splits/"
    # dict_with_results, all_test_spectra = load_all_test_results_and_test_spectra(20, test_results_folder)
    # means_and_standard_deviation = split_results_mass_all_results(dict_with_results, all_test_spectra)
    means_and_standard_deviation = load_pickled_file(os.path.join(test_results_folder, "means_and_standard_deviations_mass_bins.pickle"))

    for test_type in {"MS2Query": [],
                    "MS2Deepscore": [],
                    "MS2Deepscore 100 Da": [],
                    "Cosine score 100 Da": [],
                    "Modified cosine score 100 Da": [],
                    "Optimal": [],
                    "Random": []}:
        plot_all_with_standard_deviation_mass(
            means_and_standard_deviation,
            test_type,
            save_figure_file_name=os.path.join(
                "../../data/libraries_and_models/gnps_01_11_2022/mass_bins/", f"mass_bins_{type}.svg"))

    # from ms2query.utils import save_pickled_file
    # save_pickled_file(means_and_standard_deviation, os.path.join(test_results_folder,
    # "means_and_standard_deviations_mass_bins.pickle"))
