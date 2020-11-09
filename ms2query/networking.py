import networkx as nx
# from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np


def matches2network(query_id, matches):
    """Return a networkx Graph connecting query_id to all matches

    Args:
    ----------
    query_id: str
        ID/name of the query, will appear in the network plot
    matches: pd.DataFrame
        Library matches to the query where rows are the matches and columns are
        the features of the match.

    All data from matches is added as an edge attribute
    """
    graph = nx.Graph()  # initialise undrected graph
    lib_ids = matches.index
    # add all columns from matches to edge attributes
    att_dicts = matches.to_dict(orient='index')
    # add edges (and nodes implicitly) from query to all matches
    edge_list = []
    for lib_id in lib_ids:
        edge_list.append((query_id, lib_id, att_dicts[lib_id]))
    graph.add_edges_from(edge_list)
    return graph


def add_library_connections(graph, similarity_matrix, lib_ids):
    """Add Tanimoto similarity as edges between all library matches in graph

    Args:
    ----------
    graph: networkx Graph
        Initialised graph which contains lib_ids as nodes
    similarity_matrix: pd.DataFrame
        All vs all matrix of Tanimoto similarity between library matches
    lib_ids: list of hashable
        IDs of the items in the matrix

    Assumes that lib_ids names correspond in order to similarity_matrix
    rows/columns
    """
    matrix = np.array(similarity_matrix)
    edge_list = []
    for i, id_i in enumerate(lib_ids[:-1]):
        for j, id_j in enumerate(lib_ids[i + 1:]):
            edge_list.append((id_i, id_j, {'tanimoto': matrix[i, j + i + 1]}))
    graph.add_edges_from(edge_list)
    return graph
