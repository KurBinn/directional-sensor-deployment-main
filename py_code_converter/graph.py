import numpy as np
import networkx as nx

def graph(pop, rc):
    """
    Create a connectivity graph based on the positions of nodes and communication radius.
    """
    n = len(pop)
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(pop[i, :2] - pop[j, :2])
            if 0 < dist <= rc:
                adj_matrix[i, j] = dist
    return nx.from_numpy_array(adj_matrix)  # Updated to use from_numpy_array
