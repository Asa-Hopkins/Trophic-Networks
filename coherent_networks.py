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
    A = nx.adjacency_matrix(G)

    # get degree sequences
    outDeg = np.fromiter([d for n, d in G.out_degree()], dtype=int)
    print(sum(outDeg == 0))
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

def draw(G):
    import matplotlib.pyplot as plt
    pos = dict()
    s = coherence_stats(G)['s']
    for a in G.nodes:
        pos[a] = (np.random.random(),s[a])
    #pos=nx.spring_layout(G,k=2,pos=pos)
    nx.draw_networkx_nodes(G, pos)
    #nx.draw_networkx_edges(G, pos, edge_color='r', arrows = True)
    plt.show()

def edges(G):
    # creates a list of nodes, with a list of incoming and outgoing edges.
    # List is sorted by tropic level, so feeding forward and backwards propagation are easier.
    edges = []
    s = coherence_stats(G)['s']
    level = np.argsort(s)
    x = np.array(G.edges)
    for i in level:
        edges.append([np.where(level == i)[0][0],
                      [np.where(level == a)[0][0] for a in x[x[:,1]==i][:,0]]
                      ,[np.where(level == a)[0][0] for a in x[x[:,0]==i][:,1]]])
    edges = np.array(edges, dtype=object)#[level]
    layers = [0] #Cut nodes into sections, with nodes within a given section being independent
    #The algorithm used is simple, and a more sophisticated algorithm may be able to give more efficient sectioning
    temp = set()
    for i in edges:
        if i[0] in temp:
            layers.append(i[0])
            temp = set()
        temp = temp.union(i[2])
    return edges, layers, level

def analyse(x):
    #x is an AI.network object
    G = nx.DiGraph()
    G.add_nodes_from(range(0,sum(x.layout)))
    for n in range(0,len(x.layout)-1):
        for k in x.incoherence[n]:
            temp = np.nonzero(x.W[n])
            G.add_edges_from(zip(*(temp[0]+sum(x.layout[:n]),temp[1]+sum(x.layout[:n+k]))))
    return coherence_stats(G)["q"]
    
#G = coherent_graph(B=10, N=100, L=200, T=0.5);x,y,z = edges(G); s=coherence_stats(G)

def connectivity(length, style, params, includeInput = False):
    #A helper function for generating connectivity arrays in preset styles
    length -= 1 #exclude output layer
    styles = {"All to All":0, "ResNetX":1, "MaxDist1":2, "MaxDist2":3}
    if type(style) == type("string"):
        style = styles[style]
    con = []
    
    if style == 0:
        for a in range(length,0,-1):
            con.append(list(range(1,a+1)))
            
    elif style == 1:
        t = params[0]
        con.append([1])
        for l in range(1,length):
            if l < t:
                i = 1
            else:
                i2 = l % (2*(t - 1))
                if i2 >= 1 and i2 <= t-1:
                    i = 2*i2
                else:
                    i3 = (i2 + t - 1)%(2 * (t - 1))
                    i = 2*i3
            con[-i].append(i)
            con.append([1])
                
    elif style == 2: #All layers connect to the output layer
        for a in range(0,length):
            if length-a != 1:
                con.append([1,length-a])
            else:
                con.append([1])

    elif style == 3: #All layers connect to a "relay" layer, which then connects to the output
        for a in range(0,length):
            temp = (length - a)%params[0]
            if temp > 1:
                con.append([1,temp])
            elif temp == 1:
                con.append([1])
            else:
                con.append([1,length-a])
                
    
    if not includeInput:
        con[0] = [1]
    return con

