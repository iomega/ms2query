import os
from matplotlib.figure import Figure
from networkx import Graph
import pandas as pd
import plotly.graph_objects as go
from ms2query.networking import matches2network
from ms2query.networking import add_library_connections
from ms2query.networking import do_networking
from ms2query.networking import plot_network
from ms2query.utils import json_loader
from ms2query.s2v_functions import process_spectrums


def test_matches2network():
    """Test matches2network"""
    path_tests = os.path.dirname(__file__)
    test_matches_file = os.path.join(path_tests, "test_found_matches.csv")
    test_matches = pd.read_csv(test_matches_file, index_col=0)
    query_name = "query"
    test_network = matches2network(query_name, test_matches)
    assert isinstance(test_network, Graph), "Expected output to be nx.Graph"
    assert query_name in test_network.nodes, \
        "Expected query_id to be in the Graph"
    assert test_network.number_of_edges() == test_matches.shape[0], \
        "Expected number of edges to be equal to amount of library matches"


def test_add_library_connections():
    """Test add_library_connections"""
    path_tests = os.path.dirname(__file__)
    test_matches_file = os.path.join(path_tests, "test_found_matches.csv")
    test_matches = pd.read_csv(test_matches_file, index_col=0)
    test_matches_sim_matrix_file = os.path.join(
        path_tests, "test_found_matches_similarity_matrix.csv")
    test_sim_matrix = pd.read_csv(test_matches_sim_matrix_file, index_col=0)
    query_name = "query"
    test_network = matches2network(query_name, test_matches)
    test_network_lib_connect = add_library_connections(test_network,
                                                       test_sim_matrix,
                                                       test_matches.index)
    assert isinstance(test_network_lib_connect[test_matches.iloc[0].name][
        test_matches.iloc[1].name]["tanimoto"], float), \
        "Expected an edge with tanimoto score between these library hits"


def test_do_networking():
    """Test do_networking"""
    path_tests = os.path.dirname(__file__)
    testfile_l = os.path.join(path_tests, "testspectrum_library.json")
    spectrums_l = json_loader(open(testfile_l))
    documents_l = process_spectrums(spectrums_l)
    test_matches_file = os.path.join(path_tests, "test_found_matches.csv")
    test_matches = pd.read_csv(test_matches_file, index_col=0)
    test_matches_sim_matrix_file = os.path.join(
        path_tests, "test_found_matches_similarity_matrix.csv")
    test_sim_matrix = pd.read_csv(test_matches_sim_matrix_file, index_col=0)
    query_name = "query"
    test_network = do_networking(query_name, test_matches, test_sim_matrix,
                                 documents_l)
    assert isinstance(test_network, go.Figure), "Expected output to be Figure"


def test_plot_network():
    """Test plot_network"""
    path_tests = os.path.dirname(__file__)
    test_matches_file = os.path.join(path_tests, "test_found_matches.csv")
    test_matches = pd.read_csv(test_matches_file, index_col=0)
    test_matches_sim_matrix_file = os.path.join(
        path_tests, "test_found_matches_similarity_matrix.csv")
    test_sim_matrix = pd.read_csv(test_matches_sim_matrix_file, index_col=0)
    query_name = "query"
    test_network = do_networking(query_name, test_matches, test_sim_matrix)
    test_fig = plot_network(test_network)
    assert isinstance(test_fig, Figure), "Expected output to be Figure"
