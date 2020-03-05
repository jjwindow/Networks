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
from logbin import logbin
import matplotlib.pyplot as plt
import tqdm

# ------------------------NETWORK CLASS-----------------------------

class TheGraph:
# ---------------------------------- Initial graphs ------------------------------------------

    @staticmethod
    @numba.njit()
    def _circle_(nodes, n, k_sum, m = None):
        """
        TO BE USED ONLY BY initialGraph()

        Inserts node into circular graph
        i.e - adds node into nodes with degree 2.

        PARAMS

        nodes   :   numpy.array, corresponds to self.nodes. Array of length N nodes
                    with nodes[i] = degree of ith node.
        n       :   int, index of node to add into circle.
        k_sum   :   int, running sum of degrees.
        m       :   None, included for parameterisation of initialGraph() func. Can be 
                    passed as arg but not used here since m is fixed at 2.
        """

        nodes[n] = 2                        # Add node to circle
        k_sum += 2                          # Keep running sum of degrees
        n += 1
        return nodes, n, k_sum

    @staticmethod
    @numba.njit()
    def _er_(nodes, n, k_sum, m):
        """
        Adds node to Erdos-Renyi graph.

        PARAMS

        nodes   :   numpy.array, corresponds to self.nodes. Array of length N nodes
                    with nodes[i] = degree of ith node.
        n       :   int, index of next node in nodes array to add to graph.
        k_sum   :   int, running sum of degrees.
        m       :   number of edges to add to new node.
        """
        _m = 0                              # count up to m vertices
        nodes[n] = int(m)                   # new vertex has m edges
        attached = np.empty_like(m)         # cannot be zeros since 0 is a node in the graph
        while _m < m:
            nextNode = np.random.choice(n)  # randomly select next node from list
            if nextNode not in attached:    # check edge does not already exist
                _m += 1                     # increase m counter
                attached[_m] = nextNode
                nodes[nextNode] += 1

        k_sum += 2*m                        # degree total increases
        n += 1                              # one more node in network
        return nodes, n, k_sum

    @staticmethod
    @numba.njit()
    def _ba_(nodes, n, k_sum, m):
        """
        FOR USE BY self.growBy1() ONLY

        Grow network by 1 vertex using BA algorithm.
        Requires:   list of node degrees, number of nodes in network, running sum of
                    degrees and number of edges to attach.

        Returns:    Updated vertices, number of nodes and running k_sum.
        """
        _m = 0                              # count up to m vertices
        nodes[n] = int(m)                   # new vertex has m edges
        while _m < m:
            nextNode = np.random.choice(n)  # randomly select next node from list
            p = nodes[nextNode]/k_sum       # calc probability
            rand = np.random.random()
            if p >= rand:                   # select node according to degree
                nodes[nextNode] = int(nodes[nextNode]+1)        # if selected, increase degree by 1
                _m += 1                     # increase m counter
                k_sum += 1                  # degree total increases
        n += 1                              # one more node in network
        return nodes, n, k_sum
# ---------------------------------- __init__ ------------------------------------------
    def __init__(self, m, N, gtype = 'ba', initial = 'c', n_0 = 100):
        """
        Custom class for a growing network.

        PARAMS

        m       :   int, number of edges added to each new vertex.
        N       :   int, number of nodes in network.
        gtype   :   str, type of graph to be implemented. 
                    - 'ba' Creates a pure preferential attachment network.
                    - 'er' Creates a pure random attachment network.
        initial :   str, type of initial graph to be implemented. Only applicable for
                    BA network.
                    - 'c' Creates a circular initial graph.
                    - 'er' Creates a random initial graph.
        n_0     :   int, 100 by default.
                    Number of nodes in initial graph. Only applicable to BA network.
                    Should be >> m.
        """
        self.m = m                  # Num edges added at each iteration
        self.N = N                  # Num total vertices desired
        self.n = 0                  # Num vertices in graph at any time
        self.k_sum = 0              # Running sum of all node degrees
        self.n_0 = n_0              # Number of nodes in initial graph
        # Determine function for initial graph
        if initial == 'c':
            self._initial_ = self._circle_
        elif initial == 'er':
            self._initial_ = self._er_
        # Determine function for overall graph
        self.gtype = gtype
        if gtype == 'ba':
            self._addNode_ = self._ba_
        elif gtype == 'er':
            self._addNode_ = self._er_

        # Initialise array of degrees - ordered by chronology of node additions.
        self.nodes = np.zeros_like([0 for i in range(N)], dtype = 'int')
        return None

# ------------------------------- Make initial graph ---------------------------------------
    def initialGraph(self):
        """
        USER CALL FUNCTION
        """
        if set(self.nodes) != set([0]):
            # Case that the network has already been grown
            raise Exception("Network is not empty so cannot be initialised again.")

        m = self.m
        # Make initial network, update state variables in self
        while self.n < self.n_0:
            # Retrieve class attributes from self
            n = self.n
            nodes = self.nodes
            k_sum = self.k_sum
            self.nodes, self.n, self.k_sum = self._initial_(nodes, n, k_sum, m)
        return None
# ---------------------------------- Adding nodes ------------------------------------------
    def addNode(self):
        """
        Adds a single node to network.

        Retrieves network state variables from self and passes them into _addNode_
        which grows the network. This is done because _addNode_ is a static method
        for @numba acceleration.

        Updates the state variables in self from returns of _addNode_.
        """
        # Retrieve network state attributes
        nodes = self.nodes
        k_sum = self.k_sum
        m = self.m
        n = self.n
        # Pass into numba-enabled func, update state attributes
        self.nodes, self.n, self.k_sum = self._addNode_(nodes, n, k_sum, m)
        return None

    def growToN(self):
        """
        Grows network to final state.
        """
        with tqdm.tqdm(total=self.N, desc = f"NETWORK N: {self.N}", initial = self.n_0) as bar:
            while self.n < self.N:
                self.addNode()
                bar.update()
            return None

# ------------------------------------- Plotting ----------------------------------------------
    def plotDegree(self, plot = True):
        """
        Plot degree against frequency using logbin
        """
        k, freq = logbin(self.nodes)
        plt.grid()
        plt.ylabel('Frequency')
        plt.xlabel('Degree')
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(k, freq, 'x', label = f'm = {self.m}')
        if plot:
            plt.legend()
            plt.show()
        return None
# --------------------------- Retrieve Graph Degrees -----------------------------------
    def getAllDegrees(self):
        """
        Print self.nodes, nparray of N items where node[i] = k(i).
        """
        print("LIST OF NODE DEGREES\n", self.nodes)
        return self.nodes

# ------------------------------ Get from self -----------------------------------

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