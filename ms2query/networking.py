import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


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


def do_networking(query_id, matches, similarity_matrix):
    """Wrapper function to make and use the network, returns network (nx.Graph)

    Args:
    ----------
    query_id: str
        ID/name of the query, will appear in the network plot
    matches: pd.DataFrame
        Library matches to the query where rows are the matches and columns are
        the features of the match.
    similarity_matrix: pd.DataFrame
        All vs all matrix of Tanimoto similarity between library matches
    """
    init_network = matches2network(query_id, matches)
    # make sure to change this with how the similarity matrix gets passed
    lib_ids = matches.index.tolist()
    network = add_library_connections(init_network, similarity_matrix, lib_ids)
    return network


def plot_network(network, attribute_key='s2v_score', cutoff=0.4,
                 tan_cutoff=0.6, node_labels=False, k=1, seed=42):
    """Plot network

    Args:
    -------

    """
    width_default = 3

    # making selection based on attribute cutoffs
    library_edges = [(u, v, d) for u, v, d in network.edges(data=True) if
                     'tanimoto' in d]
    library_edges = [(u, v, d) for u, v, d in library_edges if
                     d['tanimoto'] >= tan_cutoff]
    query_edges = [(u, v, d) for u, v, d in network.edges(data=True) if
                   'tanimoto' not in d and d[attribute_key] >= cutoff]
    if not query_edges:
        print('No matches above cutoff.')
        return
    network_sub = nx.Graph(library_edges + query_edges)

    # init plot info
    fig, ax = plt.subplots()
    # pos = graphviz_layout(network_sub, prog="neato")
    pos = nx.spring_layout(network_sub, k=k, iterations=1000, seed=seed)

    nx.draw_networkx_nodes(network_sub, pos)
    q_node = \
        [node for node in network_sub.nodes if
         isinstance(node, str) and 'query' in node][
            0]
    cmap = cm.get_cmap('Reds', 100)
    # give query node darkest colour
    darkest = cmap(1.0)
    nx.draw_networkx_nodes(network_sub, pos, nodelist=[q_node],
                           node_color=[darkest])

    attr_labels = nx.get_edge_attributes(network_sub, attribute_key)
    for edge in attr_labels.keys():
        # introduce cutoff and multiply with width multiplier
        val = attr_labels[edge]
        if val > cutoff:
            width = val * width_default
            nx.draw_networkx_edges(network_sub, pos, edgelist=[edge],
                                   width=width, edge_color=cmap(val))

    tan_labels = nx.get_edge_attributes(network_sub, 'tanimoto')
    for edge in tan_labels.keys():
        # introduce cutoff and multiply with width multiplier
        val = tan_labels[edge]
        if val > tan_cutoff:
            width = val * width_default
            nx.draw_networkx_edges(network_sub, pos, edgelist=[edge],
                                   width=width / 2, style="dashed")
    if node_labels:
        nx.draw_networkx_labels(network_sub, pos, font_size=5)
    plt.axis('off')

    return fig
