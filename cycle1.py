# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx>=3.4"], stdout=subprocess.DEVNULL)

import networkx as nx, numpy as np

def WeightedDiGraph(*edges: list[tuple[int,int,float]])->nx.DiGraph:
    """
    A shorthand function for quickly generating a directed graph with weights on the edges

    >>> G = WeightedDiGraph([0,1,55],[1,2,66],[2,0,77])
    >>> G.edges[0,1]
    {'weight': 55}
    """
    return nx.DiGraph( [(u,v,{"weight":w}) for u,v,w in edges])
    

def has_cycle1(graph: nx.DiGraph)->bool:
    """
    return True iff the given graph has a directed cycle in which the product of weights is smaller than 1.

    Product < 1  <=>  sum(log(weights)) < 0, so reduce to negative-cycle detection via Bellman-Ford.
    Start all distances at 0 (virtual super-source), relax V-1 times, then check for further relaxation.

    >>> has_cycle1(WeightedDiGraph())    # empty graph
    False
    >>> has_cycle1(WeightedDiGraph([0,1,55],[1,2,66],[2,0,77]))
    False
    >>> has_cycle1(WeightedDiGraph([0,1,0.55],[1,2,0.66],[2,0,0.77]))
    True
    """
    nodes = list(graph.nodes())
    if not nodes:
        return False

    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    us, vs, ws = [], [], []
    for u, v, data in graph.edges(data=True):
        w = data.get('weight', 1.0)
        if w > 0:
            us.append(node_idx[u])
            vs.append(node_idx[v])
            ws.append(np.log(w))

    if not us:
        return False

    us_arr = np.array(us, dtype=np.int32)
    vs_arr = np.array(vs, dtype=np.int32)
    ws_arr = np.array(ws, dtype=np.float64)

    dist = np.zeros(n, dtype=np.float64)

    for _ in range(n - 1):
        new_dist = dist.copy()
        np.minimum.at(new_dist, vs_arr, dist[us_arr] + ws_arr)
        if np.array_equal(new_dist, dist):
            return False  # converged early, no negative cycle
        dist = new_dist

    # One extra relaxation: any improvement means a negative cycle exists
    check = dist.copy()
    np.minimum.at(check, vs_arr, dist[us_arr] + ws_arr)
    return bool(np.any(check < dist))


if __name__ == '__main__':
    # edges = eval(input())
    # graph = WeightedDiGraph(*edges)
    # print(has_cycle1(graph))
    import doctest
    print (doctest.testmod())
