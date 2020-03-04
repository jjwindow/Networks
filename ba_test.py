"""
Testing implementation of Barabasi-Albert network (preferential attachment)

J. J. Window
Imperial College London
Complexity & Networks - Networks Project
"""
import numpy as np
import networkx as nx
import random as r
import copy
import os
import numba

def ba_er_test(m):
    # Initialise connected ER graph (n_0 = 10)
    n_0 = 10
    G_ba_0 = nx.erdos_renyi_graph(n_0, 0.1)

    # prob P propto k
    # could use G_ba_0.degree(node) to assess prob?
    # Likely slower than using attachment list

    attach = [v[i] for i in range(2) for v in nx.edges(G_ba_0)]
    print(attach) # List of vertices with edges.
    # Randomly selecting vertices from this list leads to preferential attachment as there will
    # be more entries for vertices with more edges.

    N = n_0 + 20 # Grow graph by 20 nodes
    n = copy.copy(n_0) + 1

    while n < N:
        G_ba_0.add_node(n)
        _m = 0
        while _m < m:
            v_attach = r.choice(attach)
            G_ba_0.add_edge(n, v_attach)
            attach.append(n)
            attach.append(v_attach)
            print(f"Adding edge from {n} to {v_attach}")
            m += 1
        n += 1

def pure_ba():
    G_ba_1 = nx.Graph()
    G_ba_1.add_nodes_from([0,1])

#-------------------------------------------------------------------
# NEW CLASS

class BA_net:
    def __init__(self, m, N):
        """
        Custom class for a growing Barabasi-Albert preferential Attachment network.
        Network grows to N vertices, joining m edges at each vertex addition.
        """
        self.m = m      # Num edges added at each iteration
        self.N = N      # Num total vertices desired
        self.n = 0      # Num vertices in graph at any time
        self.k_sum = 0  # Running sum of all node degrees
        # Initialise array of degrees - ordered by chronology of node additions.
        self.nodes = np.zeros_like([0 for i in range(N)], dtype = 'float')
        return None

    @staticmethod
    @numba.njit()
    def _circle_(nodes, n_0, k_sum):
        """
        Generates initial network with n_0 vertices.
        Initial network is circular, so all n_0 vertices have degree 2.
        """
        # Initialise graph
        _n_0 = 0
        while _n_0 < n_0:
            nodes[_n_0] = 2
            _n_0 += 1
        # Keep running sum of degrees
        k_sum += 2 * n_0
        return nodes, k_sum

    def initalGraph_circle(self, n_0):
        # Retrieve class attributes from self
        nodes = self.nodes
        k_sum = self.k_sum
        # Make initial network, update state variables in self
        self.nodes, self.k_sum = self._circle_(nodes, n_0, k_sum)
        self.n = n_0
        return None
    
    def k(self, i):
        """
        Return degree for vertex i.
        """
        return self.nodes[i]

    def getFromSelf(self, *args):
        """
        Returns desired class attributes.
        Class attributes:
        m           -   edges added at each iteration
        N           -   target size of network
        n           -   current size of network
        nodes       -   array of degrees for all vertices
        k_sum       -   returns sum of degree for all vertices

        NOT COMPATIBLE WITH NUMBA.
        """
        selfargs = {}
        if 'm' in args:
            selfargs.update({'m' : self.m})
        if 'N' in args:
            selfargs.update({'N' : self.N})
        if 'n' in args:
            selfargs.update({'n' : self.n})
        if 'k_sum' in args:
            selfargs.update({'k_sum' : self.k_sum})
        if 'nodes' in args:
            selfargs.update({'nodes' : self.nodes})
        return selfargs

    def growBy1(self):
        """
        Retrieves network state variables from self and passes them into _growBy1_
        which grows the network. This is done as _growBy1_ uses @numba so does not
        have access to self.

        Updates the state variables in self from returns of _growBy1_.
        """
        # Retrieve network state attributes
        nodes = self.nodes
        n = self.n
        k_sum = self.k_sum
        m = self.m
        # Pass into numba-enabled func, update state attributes
        self.nodes, self.n, self.k_sum = self._growBy1_(nodes, n, k_sum, m)


    @staticmethod
    @numba.njit()
    def _growBy1_(nodes, n, k_sum, m):
        """
        Grow network by 1 vertex using BA algorithm.

        Requires the list of node degrees, number of nodes in network, running sum of
        degrees and number of edges to attach.

        Returns updated vertices, number of nodes and running k_sum.
        """
        _m = 0                              # count up to m vertices
        nodes[n] = m                        # new vertex has m edges
        while _m < m:
            nextNode = np.random.choice(n)           # randomly select next node from list
            p = nodes[nextNode]/k_sum       # calc probability
            rand = np.random.random()
            if p >= rand:                   # select node according to degree
                nodes[nextNode] += 1        # if selected, increase degree by 1
                _m += 1                     # increase m counter
                k_sum += 1                  # degree total increases
        n += 1                              # one more node in network
        return nodes, n, k_sum

    def getGraph(self):
        print("LIST OF NODE DEGREES\n", self.nodes)
        return self.nodes

test = BA_net(3, 25)
test.initalGraph_circle(4)
test.getGraph()  

i = 0
while i<25:
    test.growBy1()
    test.getGraph()
    i+=1
