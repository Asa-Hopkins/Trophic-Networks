"""
Library for generating and studying trophically coherent random networks.
"""

import numpy as np
import networkx as nx
from networkx.utils import not_implemented_for
from scipy import sparse
from scipy.sparse.linalg import *


def _link_prob(s1, s2, T):
    """
    Return link probability between trophic levels s1 and s2 given
    temperature T.
    """

    return np.exp(-np.abs(s2-s1-1)**2/(2*T**2))


def _create_link_probs(s, T):
    """
    Return a 2D array of link probabilities given the trophic levels s in a
    tree network.
    """

    # create mesh and use matrix indexing for adjacency purposes later
    s1, s2 = np.meshgrid(s, s, indexing='ij')

    # calculate link probabilites
    probs = _link_prob(s1, s2, T)

    return probs


def coherent_graph(B, N, L, T):
    """
    Return a random coherent directed graph G.
    Parameters
    ----------
    B : int
        Number of basal nodes with no incoming links.
    N : int
        Number of total nodes
    L : int
        Expected number of links in the finished graph.
    T : float
        Free temperature parameter, tunes the degree of trophic coherence
        in the finished graph.
    Returns
    -------
    G : networkx.DiGraph()
        A directed network with B basal nodes, N total nodes, L links (in
        expected value) tuned to have trophic coherence in accordance to
        the free temperature parameter T.
    References
    ----------
    [1] J. Klaise and S. Johnson, "From neurons to epidemics: How trophic
    coherence affects spreading processes", Chaos 26, 065310 (2016).
    Examples
    ----------
    >>> G = coherent_graph(B=10, N=100, L=450, T=0.5)
    """

    # create graph, add basal nodes and basal trophic levels
    tree = nx.DiGraph()
    tree.add_nodes_from(range(0, B))
    nx.set_node_attributes(tree, 0,'s')  # 0 OR 1
    # add nodes and single links one at a time
    for node in range(B, N):
        randNode = np.random.choice(tree.nodes())
        tree.add_edge(randNode, node, weight=1)

        # add temporary trophic level
        tree.nodes[node]['s'] = tree.nodes[randNode]['s']+1

    # create link probabilities
    s = np.fromiter(nx.get_node_attributes(tree, 's').values(),
                    dtype=float)

    probs = _create_link_probs(s, T)

    # assign zero probability to self-links
    np.fill_diagonal(probs, 0)

    # assign zero probability to links already made in the tree-stage
    probs[tuple(zip(*tree.edges()))] = 0

    # assign zero probability to links to basal species
    probs[:, 0:B] = 0

    # normalize them
    edgesToBuild = L-N+B
    sumProbs = probs.sum()
    normConst = sumProbs/edgesToBuild
    probs = probs/normConst

    # draw random numbers
    r = np.random.rand(probs.shape[0], probs.shape[1])

    # define edges to be made
    edges = probs > r
    
    # convert to a graph
    G = nx.from_numpy_matrix(edges, create_using=nx.DiGraph())

    # add original tree-links back in, use attribute 'tree' for these edges
    G.add_edges_from(tree.edges(), tree=True)

    # set trophic levels back in
    nx.set_node_attributes(G, nx.get_node_attributes(tree, 's'), 's')
    nx.set_edge_attributes(G, 1, 'weight')

    # add graph attribute normConst
    G.graph['normC'] = normConst

    return G


@not_implemented_for('undirected')
def coherence_stats(G):
    """
    Return a dictionary of coherence statistics.
    Parameters
    ----------
    G : networkx.DiGraph()
        Input network.
    Returns
    -------
    stats : dict()
        A dictionary of coherence statistics. Includes the keys "q", "s", "x"
        and "b" where
        stats["q"] : float
            Trophic incoherence parameter.
        stats["s"] : numpy.ndarray
            Array of trophic levels.
        stats["x"] : numpy.ndarray
            Array of trophic distances.
        stats["b"] : int
            Number of basal nodes with no incoming links.
    Examples
    --------
    >>> G = coherent_graph(B=10, N=100, L=450, T=0.5)
    >>> stats = coherence_stats(G)
    >>> stats["x"].mean()
    1.0
    """

    # dictionary of coherence statistics to be returned
    stats = dict.fromkeys(["q", "s", "x", "b"], None)

    # get the adjacency matrix
    A = nx.adj_matrix(G)

    # get degree sequences
    outDeg = np.fromiter([d for n, d in G.out_degree()], dtype=int)
    inDeg = np.fromiter([d for n, d in G.in_degree()], dtype=int)

    # if no basal nodes found, return NaN
    nnz = np.count_nonzero(inDeg)
    if nnz == len(inDeg):
        q = np.nan
        s = np.nan
        x = np.nan
        b = 0

    # otherwise proceed as normal
    else:
        # number of basal nodes
        b = len(inDeg) - nnz

        # define the linear system
        v = np.maximum(inDeg, 1)
        lam = sparse.diags(v, 0, dtype=int) - A

        # solve for trophic levels
        s = spsolve(lam.T, v)  # transpose

        # set them as attributes
        nx.set_node_attributes(G, dict(zip(G.nodes(), s)),'s')

        # calculate and set trophic distances
        for e in G.edges(data=True):
            G[e[0]][e[1]]['x'] = G.nodes[e[1]]['s']-G.nodes[e[0]]['s']
        # calculate q as std of x
        
        x = np.asarray([a for a in nx.get_edge_attributes(G, 'x').values()])
        mean_x = np.mean(x)
        q = np.std(x, ddof=1)  # sample variance

    # put everything in a dictionary
    stats["q"] = q
    stats["s"] = s
    stats["x"] = x
    stats["b"] = b

    return stats

##import matplotlib.pyplot as plt
##def draw(G):
##    pos = dict()
##    s = coherence_stats(G)['s']
##    for a in G.nodes:
##        pos[a] = (np.random.random(),s[a])
##    #pos=nx.spring_layout(G,k=2,pos=pos)
##    nx.draw_networkx_nodes(G, pos)
##    nx.draw_networkx_edges(G, pos, edge_color='r', arrows = True)
##    plt.show()

def edges(G):
    # creates a list of nodes, with a list of incoming and outgoing edges.
    # List is sorted by tropic level, so feeding forward and backwards propagation are easier.
    edges = []
    x = np.array(G.edges)
    for i in G.nodes:
        # represent each edge by a single integer, so (x,y) becomes x*nodes+y
        edges.append([i,x[x[:,0]==i][:,1],x[x[:,1]==i][:,0]])
    edges = np.array(edges, dtype=object)[np.argsort(coherence_stats(G)['s'])]
    layers = [0] #Cut nodes into sections, with nodes within a given section being independent
    test = []
    #The algorithm used is simple, and a more sophisticated algorithm may be able to give more efficient sectioning
    temp = set()
    for n,i in enumerate(edges):
        if i[0] in temp:
            layers.append(n)
            test.append(len(temp))
            temp = set()
        temp = temp.union(i[1])

#G = coherent_graph(B=10, N=100, L=200, T=0.5);x,y = edges(G); s=coherence_stats(G)
    return edges, layers, test
