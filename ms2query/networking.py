import networkx as nx
# from networkx.drawing.nx_pydot import graphviz_layout


def matches2network(query_id, matches):
    """Return a networkx graph connecting query_id to all matches

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
