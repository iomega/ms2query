from typing import Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import networkx as nx
import plotly.graph_objects as go
from spec2vec import SpectrumDocument


# pylint: disable=too-many-arguments,too-many-locals


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


def do_networking(query_id: str,
                  matches: pd.DataFrame,
                  similarity_matrix: pd.DataFrame,
                  library_documents: List[SpectrumDocument],
                  attribute_key: str = 's2v_score',
                  cutoff: Union[int, float] = 0.4,
                  tan_cutoff: Union[int, float] = 0.6) -> go.Figure:
    """Wrapper function to make and use the network, returns go.Figure

    Args:
    ----------
    query_id:
        ID/name of the query, will appear in the network plot
    matches:
        Library matches to the query where rows are the matches and columns are
        the features of the match.
    similarity_matrix:
        All vs all matrix of Tanimoto similarity between library matches
    library_documents:
        The library spectra as documents, indices correspond to matches index
        names.
    attribute_key:
        One of the columns in the matches dataframe
    cutoff:
        Cutoff for the attribute_key column values
    tan_cutoff:
        Cutoff for tanimoto score
    """
    # pylint: disable=protected-access
    init_network = matches2network(query_id, matches)
    # make sure to change this with how the similarity matrix gets passed
    lib_ids = matches.index.tolist()
    network = add_library_connections(init_network, similarity_matrix, lib_ids)
    # for now only get spectrumid, compound_name as node info
    node_labs = {"query": ["", ""]}  # for query node - always first in .nodes
    for lib_id in list(network.nodes)[1:]:
        node_labs[lib_id] = [
            library_documents[lib_id]._obj.get("spectrumid"),
            library_documents[lib_id]._obj.get("compound_name")]
    # make network plot
    fig = plotly_network(network, node_labs, attribute_key, cutoff, tan_cutoff)
    return fig


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
    draw_edges(network_sub, pos, 'tanimoto', tan_cutoff, width_default / 2,
               "dashed", edge_labels=edge_labels)

    if node_labels:
        nx.draw_networkx_labels(network_sub, pos, font_size=f_size)

    return fig


def plotly_network(network: nx.Graph, node_lab_dict: dict,
                   attribute_key: str = 's2v_score',
                   cutoff: Union[int, float] = 0.4,
                   tan_cutoff: Union[int, float] = 0.6,
                   k: Union[int, float] = 1, seed: int = 42,
                   width_default: Union[int, float] = 3) -> go.Figure:
    """Make a plotly plot of the network, Returns go.Figure

    Args:
    -------
    node_lab_dict:
        Dict of {node: [info]}, where info are strings and all info lists are
        equal in size
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
    network.add_node(q_node)  # make sure query is always in the plot

    pos = nx.spring_layout(network, k=k, iterations=500, seed=seed)

    edge_trace = []
    for u, v, d in library_edges:
        edge, edge_text = make_plotly_edge(u, v, d, pos, 'tanimoto',
                                           width_default * 0.8, "dash",
                                           max_val=1)
        edge_trace.append(edge)
        edge_trace.append(edge_text)

    max_val = max([e[2][attribute_key] for e in query_edges])
    if max_val < 1:
        max_val = 1  # to keep 1 always the max value for scores: s2v, cosine..
    red_cmap = cm.get_cmap('Reds')
    for u, v, d in query_edges:
        edge, edge_text = make_plotly_edge(u, v, d, pos, attribute_key,
                                           width_default, "solid", max_val,
                                           red_cmap)
        edge_trace.append(edge)
        edge_trace.append(edge_text)

    node_x = []
    node_y = []
    node_type = []
    for node in network.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        type_val = 0
        if isinstance(node, str):
            if "query" in node:
                type_val = 1
        node_type.append(type_val)

    node_label = []
    custom_lab = []
    for node, _ in enumerate(network.adjacency()):
        # node_adjacencies.append(len(adjacencies[1]))
        # append(node_lab_dict[node]
        lab_name = list(network.nodes)[node]
        node_label.append(lab_name)
        custom_lab.append(node_lab_dict[lab_name])
    custom_str = ''.join([f"<br>%{{customdata[{i}]}}" for i in
                          range(len(custom_lab[0]))])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        text=node_label,
        customdata=custom_lab,
        hovertemplate="%{text}"+custom_str+"<extra></extra>",
        marker=dict(
            size=40,
            color=np.array(node_type).astype(int),
            colorscale='portland',
            line_color="white",
            line_width=2))

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

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=layout)
    return fig


def make_plotly_edge(u: Union[str, int], v: Union[str, int], d: dict,
                     pos: dict, attribute: str, width_default: float,
                     style: str = "solid", max_val: Union[float, int] = 1,
                     cmap: Union[None, colors.Colormap] = None):
    """Return go.Scatter for the edge object

    Args:
    -------
    u:
        Starting node of edge.
    v:
        End node of edge.
    d:
        Attribute data of node
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
    max_val:
        Maximum value of edge attribute.
    cmap: matplotlib.colors.colourmap, optional
        If provided, a cmap to colour the edges by attribute score.
        Default = none
    """
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    xs = (x0, x1, None)
    ys = (y0, y1, None)
    # text coordinates
    x_txt = [(x0 + x1) / 2]
    y_txt = [(y0 + y1) / 2]
    # default is the library connection parameters
    val = d[attribute]
    cor_val = val / max_val
    if isinstance(val, bool):
        val = int(val)
    e_width = width_default * cor_val
    if e_width == 0.0:
        e_width = 0.8  # for mass_match==0
    if cmap:
        e_colour = colors.to_hex(cmap(cor_val))
    else:
        e_colour = "#888"
    edge_trace = go.Scatter(
        x=xs,
        y=ys,
        line=dict(width=e_width, color=e_colour, dash=style),
        mode='lines')
    # edge_text = ["{}: {}".format(attribute, str(d[attribute]))]
    # edge_trace.text = edge_text

    txt_trace = go.Scatter(
        x=x_txt,
        y=y_txt,
        customdata=[["{}: {:.3f}".format(attribute, val)]],
        mode='text',
        hovertemplate="%{customdata[0]}<extra></extra>",
        showlegend=False)

    return edge_trace, txt_trace
