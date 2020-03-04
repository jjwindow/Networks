"""
Testing implementation of Barabasi-Albert network (preferential attachment)

J. J. Window
Imperial College London
Complexity & Networks - Networks Project
"""
# --------------------------MODULES---------------------------------
import numpy as np
import copy
import numba

# ------------------------NETWORK CLASS-----------------------------

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
        FOR USE BY self.initialGraph_circle() ONLY

        Generates initial circular network with n_0 vertices.
        i.e - first n_0 nodes gain degree 2.
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
        """
        USER CALL FUNCTION

        Calls _circle_ with class attributes and n_0 to initialise
        network. Updates node, k_sum in self after initialisation.
        """
        if set(self.nodes) != set([0]):
            # Case that the network has already been grown
            raise Exception("Network is not empty so cannot be initialised again.")
        # Retrieve class attributes from self
        nodes = self.nodes
        k_sum = self.k_sum
        # Make initial network, update state variables in self
        self.nodes, self.k_sum = self._circle_(nodes, n_0, k_sum)
        self.n = n_0
        return None

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
        Kept in for potential future reference.
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
        USER CALL FUNCTION

        Retrieves network state variables from self and passes them into _growBy1_
        which grows the network. This is done because _growBy1_ uses @numba so 
        does not have access to self.

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
        FOR USE BY self.growBy1() ONLY

        Grow network by 1 vertex using BA algorithm.
        Requires:   list of node degrees, number of nodes in network, running sum of
                    degrees and number of edges to attach.

        Returns:    Updated vertices, number of nodes and running k_sum.
        """
        _m = 0                              # count up to m vertices
        nodes[n] = m                        # new vertex has m edges
        while _m < m:
            nextNode = np.random.choice(n)  # randomly select next node from list
            p = nodes[nextNode]/k_sum       # calc probability
            rand = np.random.random()
            if p >= rand:                   # select node according to degree
                nodes[nextNode] += 1        # if selected, increase degree by 1
                _m += 1                     # increase m counter
                k_sum += 1                  # degree total increases
        n += 1                              # one more node in network
        return nodes, n, k_sum

    def getGraph(self):
        """
        Print self.nodes, nparray of N items where node[i] = k(i).
        """
        print("LIST OF NODE DEGREES\n", self.nodes)
        return self.nodes