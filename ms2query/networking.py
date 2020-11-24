from typing import Union, List
import numpy as np
import pandas as pd
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
                  similarity_matrix: Union[np.array, pd.DataFrame],
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
    # for query node - always first in .nodes
    node_labs = {"query": ["", "", ""]}
    for lib_id in list(network.nodes)[1:]:
        lib_doc = library_documents[lib_id]
        node_labs[lib_id] = [
            lib_doc._obj.get("spectrumid"),
            lib_doc._obj.get("compound_name"),
            f'm/z {lib_doc._obj.get("parent_mass"):.3f}'
        ]
    # make network plot
    fig = plotly_network(network, node_labs, attribute_key, cutoff, tan_cutoff)
    return fig


def plotly_network(network: nx.Graph,
                   node_lab_dict: dict,
                   attribute_key: str = 's2v_score',
                   cutoff: Union[int, float] = 0.4,
                   tan_cutoff: Union[int, float] = 0.6,
                   k: Union[int, float] = 1,
                   seed: int = 42,
                   width_default: Union[int, float] = 3) -> go.Figure:
    """Make a plotly plot of the network, Returns go.Figure

    Args:
    -------
    network:
        Network to plot, based on the cutoffs you provide a subset will be plot
    node_lab_dict:
        Dict of {node: [info]}, where info are strings and all info lists are
        equal in size
    attribute_key:
        One of the columns in the matches dataframe
    cutoff:
        Cutoff for the attribute_key column values
    tan_cutoff:
        Cutoff for tanimoto score
    k:
        Optimal node distance for nx.spring_layout
    seed:
        Seed used by nx.spring_layout
    width_default:
        The max width of the edges
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

    # Detect not attached nodes
    nodes_detached = []
    for node in network.nodes:
        if not str(node).startswith("query") and not nx.has_path(network, q_node, node):
            nodes_detached.append(node)

    # Update network
    network = network.subgraph([node for node in network.nodes if node not in nodes_detached])
    library_edges = [(u, v, d) for u, v, d in network.edges(data=True) if
                      'tanimoto' in d]
    library_edges = [(u, v, d) for u, v, d in library_edges if
                      d['tanimoto'] >= tan_cutoff]
    query_edges = [(u, v, d) for u, v, d in network.edges(data=True) if
                    'tanimoto' not in d and d[attribute_key] >= cutoff]
    
    # Position nodes
    pos = nx.spring_layout(network, k=k, iterations=500, seed=seed)

    edge_trace = []
    name = "tanimoto score"
    for i, (u, v, d) in enumerate(library_edges):
        show_lab = bool(i == 0)
        edge, edge_text = make_plotly_edge(u, v, d, pos, 'tanimoto',
                                           width_default * 0.8, "dash",
                                           max_val=1, name=name, group=True,
                                           show_lab=show_lab)
        edge_trace.append(edge)
        edge_trace.append(edge_text)

    max_val = max([e[2][attribute_key] for e in query_edges])
    if max_val < 1:
        max_val = 1  # to keep 1 always the max value for scores: s2v, cosine..
    red_cmap = cm.get_cmap('Reds')
    for i, (u, v, d) in enumerate(query_edges):
        show_lab = bool(i == 0)
        edge, edge_text = make_plotly_edge(u, v, d, pos, attribute_key,
                                           width_default, "solid", max_val,
                                           red_cmap, attribute_key, True,
                                           show_lab)
        edge_trace.append(edge)
        edge_trace.append(edge_text)

    nodes_x = []
    nodes_y = []
    nodes_type = []
    for node in network.nodes():
        x, y = pos[node]
        nodes_x.append(x)
        nodes_y.append(y)
        if isinstance(node, str) and "query" in node:
            nodes_type.append("query")
        else:
            nodes_type.append("candidate")

    nodes_label = []
    custom_lab = []
    for node, _ in enumerate(network.adjacency()):
        # node_adjacencies.append(len(adjacencies[1]))
        # append(node_lab_dict[node]
        lab_name = list(network.nodes)[node]
        nodes_label.append(lab_name)
        custom_lab.append(node_lab_dict[lab_name])  # read the dict for labels
    # gather info in a df
    nodes = pd.DataFrame({"x": nodes_x,
                          "y": nodes_y,
                          "type": nodes_type,
                          "label": nodes_label,
                          "custom_label": custom_lab})

    # make string for including all node label info from dict in hovertemplate
    custom_str = ''.join([f"<br>%{{customdata[{i}]}}" for i in
                          range(len(custom_lab[0]))])

    node_trace = go.Scatter(
        x=nodes[nodes["type"] == "candidate"]["x"],
        y=nodes[nodes["type"] == "candidate"]["y"],
        mode='markers',
        text=nodes[nodes["type"] == "candidate"]["label"],
        customdata=nodes[nodes["type"] == "candidate"]["custom_label"],
        hovertemplate="%{text}"+custom_str+"<extra></extra>",
        name="found candidates",
        marker=dict(
            size=40,
            color='dodgerblue',
            line_color="white",
            line_width=2))

    node_trace_query = go.Scatter(
        x=nodes[nodes["type"] == "query"]["x"],
        y=nodes[nodes["type"] == "query"]["y"],
        mode='markers',
        text=nodes[nodes["type"] == "query"]["label"],
        customdata=nodes[nodes["type"] == "query"]["custom_label"],
        hovertemplate="%{text}"+custom_str+"<extra></extra>",
        name="query spectrum",
        marker=dict(
            size=40,
            color='crimson',
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

    fig = go.Figure(data=edge_trace + [node_trace, node_trace_query],
                    layout=layout)
    return fig


def make_plotly_edge(u: Union[str, int],
                     v: Union[str, int],
                     d: dict,
                     pos: dict,
                     attribute: str,
                     width_default: float,
                     style: str = "solid",
                     max_val: Union[float, int] = 1,
                     cmap: Union[None, colors.Colormap] = None,
                     name: str = None,
                     group: bool = None,
                     show_lab: bool = False):
    """Returns go.Scatter for the edge object

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
    width_default:
        The max width of an edge
    style:
        The style of the edges. Default = solid
    max_val:
        Maximum value of edge attribute.
    cmap:
        If provided, a cmap to colour the edges by attribute score.
        Default = none, meaning #888 will be used as a colour
    name:
        If provided, give edge a name in the legend
    group:
        Assign edge to a legendgroup - will be the attribute
    show_lab:
        Show edge in the legend
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
    if group:
        group = attribute
    edge_trace = go.Scatter(
        x=xs,
        y=ys,
        line=dict(width=e_width, color=e_colour, dash=style),
        mode='lines',
        name=name,
        showlegend=show_lab,
        legendgroup=group)

    txt_trace = go.Scatter(
        x=x_txt,
        y=y_txt,
        customdata=[["{}: {:.3f}".format(attribute, val)]],
        mode='text',
        hovertemplate="%{customdata[0]}<extra></extra>",
        showlegend=False)

    return edge_trace, txt_trace
