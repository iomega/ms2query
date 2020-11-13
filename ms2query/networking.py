import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import plotly.graph_objects as go


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
               style="solid", cmap=None, edge_labels=True):
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
    edge_labels: bool, optional
        Plot edge labels
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
            if edge_labels:
                if isinstance(val, float):
                    val = f"{val:.2f}"
                nx.draw_networkx_edge_labels(network, pos,
                                             edge_labels={edge: val},
                                             font_size=4.5)


def plot_network(network, attribute_key='s2v_score', cutoff=0.4,
                 tan_cutoff=0.6, node_labels=False, k=1, seed=42,
                 width_default=3, edge_labels=False):
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
    edge_labels: bool, optional
        Plot edge labels
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
               cmap=cmap, edge_labels=edge_labels)
    # draw tanimoto edges
    draw_edges(network_sub, pos, 'tanimoto', tan_cutoff, width_default/2,
               "dashed", edge_labels=edge_labels)

    if node_labels:
        nx.draw_networkx_labels(network_sub, pos, font_size=f_size)

    return fig


def plotly_network(network, attribute_key='s2v_score', cutoff=0.4,
                   tan_cutoff=0.6, k=1, seed=42):
    """

    Args:
    -------

    """
    # pylint: disable=too-many-locals
    library_edges = [(u, v, d) for u, v, d in network.edges(data=True) if
                     'tanimoto' in d]
    library_edges = [(u, v, d) for u, v, d in library_edges if
                     d['tanimoto'] >= tan_cutoff]
    query_edges = [(u, v, d) for u, v, d in network.edges(data=True) if
                   'tanimoto' not in d and d[attribute_key] >= cutoff]
    q_node = [node for node in network.nodes if isinstance(node, str)
              and 'query' in node][0]
    network = nx.Graph(library_edges + query_edges)

    pos = nx.spring_layout(network, k=k, iterations=500, seed=seed)

    edge_x = []
    edge_y = []
    edge_style = []
    for edge in network.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=4.0, color='#888'),
        hoverinfo='text',
        mode='lines')

    node_x = []
    node_y = []
    node_type = []
    for node in network.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_type.append(node == "query")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=40,
            color=np.array(node_type).astype(int),
            colorscale='Bluered',
            line_color="white",
            line_width=2))

    node_adjacencies = []
    node_label = []
    for node, adjacencies in enumerate(network.adjacency()):
        # node_adjacencies.append(len(adjacencies[1]))
        node_label.append(list(network.nodes)[node])

    node_trace.text = node_label

    layout = go.Layout(
        autosize=True,
        width=700,
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',

        xaxis=go.layout.XAxis({'showgrid': False,
                               'visible': False, }),
        yaxis=go.layout.YAxis({'showgrid': False,
                               'visible': False, }),
        margin=go.layout.Margin(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4,
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=layout)
    return fig
