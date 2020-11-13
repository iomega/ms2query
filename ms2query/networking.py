import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx


# pylint: disable=too-many-arguments


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
    if 'query' not in query_id:
        query_id = "query_" + query_id
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


def draw_edges(network, pos, attribute, attr_cutoff, width_default,
               style="solid", cmap=None):
    """
    Draw edges where width is determined by attribute score, if score > cutoff

    Args:
    -------
    network: nx.Graph
        Network for which to draw edges
    pos: dict of {node: [x,y]}
        Contains coordinates for each node
    attribute: str
        The attribute that should be looked at for the score
    attr_cutoff: float
        Cutoff for the attribute scores
    width_default: int
        The max width of an edge
    style: str, optional
        The style of the edges. Default = solid
    cmap: matplotlib.colourmap, optional
        If provided, a cmap to colour the edges by attribute score.
        Default = none
    """
    labels = nx.get_edge_attributes(network, attribute)
    max_val = max(labels.values())
    if max_val < 1:
        max_val = 1  # to keep 1 always the max value for scores: s2v, cosine..
    for edge in labels.keys():
        # introduce cutoff and multiply with width multiplier
        val = labels[edge]
        if val >= attr_cutoff:
            cor_val = val / max_val
            width = cor_val * width_default
            if cmap:
                nx.draw_networkx_edges(network, pos, edgelist=[edge],
                                       width=width, style=style,
                                       edge_color=cmap(cor_val))
            else:
                nx.draw_networkx_edges(network, pos, edgelist=[edge],
                                       width=width, style=style)


def plot_network(network, attribute_key='s2v_score', cutoff=0.4,
                 tan_cutoff=0.6, node_labels=False, k=1, seed=42,
                 width_default=3):
    """Plot network, Returns matplotlib.figure.Figure

    Args:
    -------
    network: nx.Graph
        Network to plot
    attribute_key: str, optional
        Name of the attribute to restrict the network on. Default = 's2v_score'
    cutoff: int/float, optional
        Cutoff to restrict the attribute key on. Default = 0.4
    tan_cutoff: float, optional
        Cutoff to restrict tanimoto score on. Default = 0.6
    node_labels: bool, optional
        Show node_labels or not. Default = False
    k: int, optional
        Optimal node distance. Default = 1
    seed: int, optional
        Seed used for spring layout. Default = 42
    width_default: int/float, optional
        Default width for the edges. Default = 3
    """
    # suppress pylint for now
    # pylint: disable=too-many-locals
    f_size = 5.5

    # making selection based on attribute cutoffs
    library_edges = [(u, v, d) for u, v, d in network.edges(data=True) if
                     'tanimoto' in d]
    library_edges = [(u, v, d) for u, v, d in library_edges if
                     d['tanimoto'] >= tan_cutoff]
    query_edges = [(u, v, d) for u, v, d in network.edges(data=True) if
                   'tanimoto' not in d and d[attribute_key] >= cutoff]
    q_node = [node for node in network.nodes if isinstance(node, str)
              and 'query' in node][0]

    # make colours for query info
    cmap = cm.get_cmap('Reds', 100)
    # give query node darkest colour
    darkest = cmap(1.0)

    # init plot
    fig, _ = plt.subplots()
    plt.axis('off')

    # plot empty network if there are no connections to query
    if not query_edges:
        empty_network = nx.Graph()
        empty_network.add_node(q_node)
        q_pos = {q_node: [0, 0]}
        nx.draw_networkx_nodes(empty_network, pos=q_pos,
                               nodelist=[q_node], node_color=[darkest])
        if node_labels:
            nx.draw_networkx_labels(empty_network, pos=q_pos, font_size=f_size)
        print('No matches above cutoff.')
        return fig

    # init plot info
    network_sub = nx.Graph(library_edges + query_edges)
    # pos = graphviz_layout(network_sub, prog="neato")
    pos = nx.spring_layout(network_sub, k=k, iterations=1000, seed=seed)

    # draw nodes
    nx.draw_networkx_nodes(network_sub, pos)
    nx.draw_networkx_nodes(network_sub, pos, nodelist=[q_node],
                           node_color=[darkest])  # give query different colour
    # draw attribute edges
    draw_edges(network_sub, pos, attribute_key, cutoff, width_default,
               cmap=cmap)
    # draw tanimoto edges
    draw_edges(network_sub, pos, 'tanimoto', tan_cutoff, width_default/2,
               "dashed")

    if node_labels:
        nx.draw_networkx_labels(network_sub, pos, font_size=f_size)

    return fig
