import networkx as nx

def connectivity_graph(G, bat_ex):
    """
    Check if the graph is connected after excluding exhausted nodes.
    """
    number_nodes = G.number_of_nodes() - len(bat_ex)
    visited_nodes = list(nx.dfs_preorder_nodes(G, source=0))
    return int(len(visited_nodes) == number_nodes)
