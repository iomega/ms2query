from ms2query.networking import matches2network
from networkx import Graph
import pandas as pd
import os


def test_matches2network():
    """Test matches2network"""
    path_tests = os.path.dirname(__file__)
    test_matches_file = os.path.join(path_tests, "test_found_matches.csv")
    test_matches = pd.read_csv(test_matches_file, index_col=0)
    query_name = "query"
    test_network = matches2network(query_name, test_matches)
    assert isinstance(test_network, Graph), "Expected output to be nx.Graph"
    assert query_name in test_network.nodes,\
        "Expected query_id to be in the Graph"
    assert test_network.number_of_edges() == test_matches.shape[0],\
        "Expected number of edges to be equal to amount of library matches"
