"""
Module containing the network class used in the 
investigation of Barabasi-Albert, Pure Random Attachment
and Random Walk graphs.

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
import os
from collections import Counter
from itertools import combinations, repeat
from math import factorial

# ------------------------NETWORK CLASS-----------------------------

class TheGraph:
# ---------------------------------- Initial graphs ------------------------------------------

    @staticmethod
    @numba.njit()
    def _circle_(nodes, rselect, edges, n_0, p_acc):
        """
        FOR USE BY self.initialGraph ONLY

        Generates an initial graph of n_0 nodes, each with degree 2 (equivalent to 
        a circular network).

        PARAMETERS
        --------------------------------------------------------------------------------
        nodes   :   numpy.array, corresponds to self.nodes. Array of length N nodes
                    with nodes[i] = degree of ith node (always 0 for initial graph).
        rselect :   numpy.array, corresponds to self.rselect. Random selection array 
                    containing node indices i repeated k_i times where k_i is the 
                    degree of the node.
                    Used to generate edges with preferential attachment.
        nn      :   numpy.2darray, array of nearest neighbours for each node in the
                    network. Only used for random walk graph.
        n_0     :   int, number of initial graph nodes
        p_acc   :   None for initialType = 'c'. Included for parameterisation of 
                    initialGraph() func.

        RETURNS
        --------------------------------------------------------------------------------
        nodes   :   numpy.array, as above but including new initial graph.
        rselect :   numpy.array, as above but including new initial graph.
        rcount  :   int, running index for rselect. Saves computation of len(rselect).
        n_0     :   int, becomes self.n when returned.
        """
        n = 0
        rcount = 0

        while n < n_0:
            nodes[n] = 2                        # Add node to circle
            rselect[rcount] = n
            rselect[rcount+1] = n
            n += 1
            rcount += 2

        edges[0] = (n_0, 1)                     # First and last edges
        edges[n_0] = (n_0-1, 0)
        i = 1
        while i < n_0:                          # Other edges in circle
            edges[i] = (i-1, i+1)
            i += 1

        return nodes, rselect, rcount, edges, n_0

    @staticmethod
    # @numba.njit()
    def _er_init_(nodes, rselect, edges, n_0, p_acc):
        """
        FOR USE BY self.initialGraph ONLY

        Generates an initial graph from the Erdos-Renyi algorithm.
        The graph is made from n_0 initial nodes, and edges are selected
        with probability p_acc from all combinations of edges.

        PARAMETERS
        --------------------------------------------------------------------------------
        nodes   :   numpy.array, corresponds to self.nodes. Array of length N nodes
                    with nodes[i] = degree of ith node (always 0 for initial graph).
        rselect :   numpy.array, corresponds to self.rselect. Random selection array 
                    containing node indices i repeated k_i times where k_i is the 
                    degree of the node.
                    Used to generate edges with preferential attachment.
        nn      :   numpy.2darray, array of nearest neighbours for each node in the
                    network. Only used for random walk graph.
        n_0     :   int, number of initial graph nodes
        p_acc   :   None for initialType = 'c'. Included for parameterisation of 
                    initialGraph() func.

        RETURNS
        --------------------------------------------------------------------------------
        nodes   :   numpy.array, as above but including new initial graph.
        rselect :   numpy.array, as above but including new initial graph.
        rcount  :   int, running index for rselect. Saves computation of len(rselect).
        n_0     :   int, becomes self.n when returned.
        """
        rcount = 0
        # Dummy list of node indices
        _nodes = np.arange(n_0, dtype = int) 
        # Select edges from all combinations with probability p_acc
        # edges = np.asarray([edge for edge in combinations(_nodes, 2) if np.random.random() < p_acc], dtype = int)
        edges = np.asarray([edge for edge in combinations(_nodes, 2) if np.random.random() < p_acc], dtype = int)
        # Make dictionary of node indices and their degrees
        degreeDict = Counter(edges.flatten())
        for n, k in degreeDict.items():
            n = int(n)
            k = int(k)
            # Populate nodes array with degrees
            nodes[n] = k
            # Add indices to selection list according to their degree
            rselect[rcount:rcount+k] = n
            rcount += k
        return nodes, rselect, rcount, edges, n_0

# ---------------------------------- Full Graphs ------------------------------------------

    @staticmethod
    @numba.njit()
    def _rnd_(nodes, n, m):
        """
        FOR USE BY self.growBy1() ONLY

        Adds a single node to the graph and connects it to m other nodes following
        the random attachment model. Selects other nodes uniformly at random.

        PARAMS
        ---------------------------------------------------------------
        nodes   :   numpy.array, degrees of all nodes in network.
        n       :   int, index of node being added.
        m       :   int, number of edges to be added.

        RETURNS
        ---------------------------------------------------------------
        nodes   :   numpy.array, as above but includes new node.
        rselect :   None, included for parameterisation of _addNode_.
        rcount  :   None, included for parameterisation of _addNode_.
        n       :   int, as above but updated (+= 1).
        nn      :   None, included for parameterisation of _addNode_.
        """
        _m = 0                              # count up to m vertices
        nodes[n] = int(m)                   # new vertex has m edges
        attached = np.array([-1 for i in range(m)], dtype = int)    # cannot be zeros since 0 is a node in the graph
        while _m < m:
            nextNode = np.random.choice(n)  # randomly select next node from list
            isAttached = False
            for att in attached:
                if att == nextNode:
                    isAttached = True
                
            if isAttached == False:         # check edge does not already exist
                attached[_m] = nextNode
                nodes[nextNode] += 1
                _m += 1                     # increase m counter
        n += 1                              # one more node in network
        rselect = None                      # Included for correct num of returns
        rcount = None
        nn = None
        return nodes, rselect, rcount, n, nn

    @staticmethod
    @numba.njit()
    def _ba_(nodes, rselect, rcount, n, m):
        """
        FOR USE BY self.growBy1() ONLY

        Adds a single node to the graph and connects it to m other nodes following
        the Barabasi-Albert model. Selects other nodes with probability proportional to
        their degree, using an attachment list.

        PARAMS
        ---------------------------------------------------------------
        nodes   :   numpy.array, degrees of all nodes in network.
        rselect :   numpy.array, contains k_i elements of node i.
        rcount  :   int, working index of rselect.
        n       :   int, index of node being added.
        m       :   int, number of edges to be added.

        RETURNS
        ---------------------------------------------------------------
        nodes   :   numpy.array, as above but includes new node.
        rselect :   numpy.array, as above but updated with node addition.
        rcount  :   int, as above but updated (+= 2*m).
        n       :   int, as above but updated (+= 1).
        nn      :   None, included for parameterisation of _addNode_.
        """
        _m = 0                              # count up to m vertices
        nodes[n] = int(m)                   # new vertex has m edges

        while _m < m:
            nextNode = np.random.choice(rselect[:rcount]) # Select next node with prob ~k
            nodes[nextNode] += 1
            rcount += 1
            rselect[rcount] = nextNode
            _m += 1                         # increase m counter

        rselect[rcount:rcount+m] = n        # Add new vertex to random select list
        rcount += m
        n += 1                              # one more node in network
        nn = None
        return nodes, rselect, rcount, n, nn

    @staticmethod
    @numba.njit()
    def _rndwlk_(nodes, nn, n, m, q):
        """
        FOR USE BY self.growBy1() ONLY

        Adds a single node to the graph and connects it to m other nodes following
        the random walk model. Selects other nodes uniformly at random, then walks
        randomly to nearest neighbours with probability q at each step.

        PARAMS
        ---------------------------------------------------------------
        nodes   :   numpy.array, degrees of all nodes in network.
        nn      :   numpy.2darray, nearest neighbours of all nodes in network.
        n       :   int, index of node being added.
        m       :   int, number of edges to be added.
        q       :   float, probability of moving to another nearest neighbour at 
                    each step during random walk. 0 <= q < 1.

        RETURNS
        ---------------------------------------------------------------
        nodes   :   numpy.array, as above but includes new node.
        rselect :   None, included for parameterisation of _addNode_.
        rcount  :   None, included for parameterisation of _addNode_.
        n       :   int, as above but updated (+= 1).
        nn      :   numpy.2darray, as above but updated with new node.
        """
        _m = 0                              # count up to m vertices
        nodes[n] = int(m)                   # new vertex has m edges
        attached = np.array([-1 for i in range(m)], dtype = int)    # cannot be zeros since 0 is a node

        while _m < m:
            nextNode = np.random.choice(n)
            isAttached = any([att == nextNode for att in attached])
            while isAttached:
                # repeat until node not already attached
                nextNode = np.random.choice(n)
                isAttached = any([att == nextNode for att in attached])

            walk = (np.random.random() <= q)
            # Go on a random walk:
            while walk:
                _nextNode = np.random.choice(nn[nextNode]) 
                isAttached = any([att == _nextNode for att in attached])
                # Filler vals in nn[nextNode] are -1, exclude these:
                while _nextNode == -1 or isAttached:
                    _nextNode = np.random.choice(nn[nextNode]) 
                nextNode = _nextNode
                walk = (np.random.random() <= q)

            attached[_m] = nextNode
            nodes[nextNode] += 1
            nn[n][_m] = nextNode
            empty = np.where(nn[nextNode]==-1)
            idx = empty[0]                  # Index of first empty element of nn[i]
            nn[nextNode][idx] = n
            _m += 1                         # increase m counter
        n += 1                              # one more node in network
        rselect = None                      # Needed for num of returns.
        rcount = None               
        return nodes, rselect, rcount, n, nn


# ---------------------------------- __init__ ------------------------------------------
    def __init__(self, m, N, gtype = 'ba', initial = 'c', n_0 = 100, p_acc = 0.2, q = 0.5, scale = 1.2):
        """
        Custom class for a growing network of different types.

        PARAMS
        --------------------------------------------------------------------------------
        m       :   int, number of edges added to each new vertex.
        N       :   int, number of nodes in network.
        gtype   :   str, type of graph to be implemented. 
                    - 'ba' Creates a pure preferential attachment network.
                    - 'rnd' Creates a pure random attachment network.
                    - 'rndwlk' Creates a random walk preferntial attachment network.
        initial :   str, type of initial graph to be implemented. Only applicable for
                    BA network.
                    - 'c' Creates a circular initial graph.
                    - 'er' Creates an ER initial graph.
        n_0     :   int, 100 by default.
                    Number of nodes in initial graph.
        p_acc   :   float, probability of selecting an edge for the ER initial graph.
        q       :   float, probability of continuing random walk at every step during
                    random walk network.
        scale   :   float, log bin scale used when plotting data.
                    A value of 1 corresponds to no binning.
        """
        if initial == 'er' and (p_acc < 0 or p_acc > 1):
            raise Exception("p_acc is a probability and must be between 0 and 1.")
        elif initial == 'rndwlk' and (q < 0 or q> 1):
            raise Exception("q is a probability and must be between 0 and 1.")
                
        self.n = 0                  # Num vertices in graph at any time
        self.rcount = 0             # Index for BA random vertex selection

        self.m = m                  # Num edges added at each iteration
        self.N = N                  # Num total vertices desired
        self.n_0 = n_0              # Number of nodes in initial graph
        self.initialType = initial  # Type of initial graph      
        self.p_acc = p_acc          # Probability of acceptance for ER initial graph
        self.q = q                  # Random walk probability
        self.scale = scale          # Log binning scale

        # Determine function for overall graph
        self.gtype = gtype
        if gtype == 'ba':
            self._addNode_ = self._ba_
        elif gtype == 'rnd':
            self._addNode_ = self._rnd_
        elif gtype == 'rndwlk':
            self._addNode_ = self._rndwlk_
        else:
            raise Exception("Specify graph type from one of 'ba', 'rnd' or 'rndwlk'.")
        # Determine function for initial graph
        if initial == 'c':
            self._initial_ = self._circle_
        elif initial == 'er':
            self._initial_ = self._er_init_
        else:
            raise Exception("Specify initial graph type from one of 'ba' or 'er'.")

        self.nodes = np.array([0 for i in range(N)], dtype = 'int')         # Array of degrees
        if gtype == 'rndwlk':
            # Array of arrays of nearest neighbours.
            self.nn = np.array([np.array([-1 for i in range(int(N/10))], dtype = 'int') for i in range(N)])
        # Attachment array for node selection (BA)
        self.rselect = np.array([-1 for i in range(4*m*N)], dtype = 'int')
        # -1 because +ve ints correspond to nodes
        # 4mN large enough for initial graph + extra 2*m*N degrees from graph
        edgenum = int(factorial(n_0)/(2*factorial(n_0-2))) # Max number of edges in ER graph
        self.edges = np.asarray([(0,0) for i in range(edgenum)]) # empty edge list
        return None

# ------------------------------- Make initial graph ---------------------------------------
    def initialGraph(self):
        """
        CALLER METHOD

        Instantiates the initial graph specified by the caller. Takes no arguments
        but passes values from self into one of the numba-enabled initial graph functions,
        _circle_ and _er_. The values returned from the initial graph function are stored 
        again in self.
        """
        if set(self.nodes) != set([0]):
            # Case that the network has already been grown
            raise Exception("Network is not empty so cannot be initialised again.")
        
        # Retrieve class attributes from self
        nodes = self.nodes
        p_acc = self.p_acc
        n_0 = self.n_0
        rselect = self.rselect
        edges = self.edges

        self.nodes, self.rselect, self.rcount, self.edges, self.n = self._initial_(nodes, rselect, edges, n_0, k_sum, p_acc)
        if self.initialType == 'rndwlk':
            edges = self.edges
            nn = self.nn
            self.nn = self.edgesToNeighbours(edges, nn)
        return None

    @staticmethod
    @numba.njit()
    def edgesToNeighbours(edges, nn):
        """
        Maps edge list from initial graph into array of nearest neighbours used to grow graph in
        random walk model.

        PARAMS
        ---------------------------------------------------------------------
        edges   :   numpy.array, array of tuples containing pairs of nodes connected by edges.
        nn      :   numpy.2darray, every ith element contains array of nearest neighbours of
                    node i.

        RETURNS
        ---------------------------------------------------------------------
        nn      :   numpy.2darray, same as above but populated with data from initial graphs.
        """
        # Remove empty edges from list
        edges = np.asarray([e for e in edges if e != (0,0)])
        # Add edges to nearest neighbours list
        for n1, n2 in edges:
            empty1 = np.where(nn[n1] == -1)
            empty2 = np.where(nn[n2] == -1)
            # First 'empty' elements of nn[n1,2].
            idx1 = empty1[0]                      
            idx2 = empty2[0]
            # Add n1,2 as neighbour of n2,1
            nn[n1][idx1] = n2
            nn[n2][idx2] = n1
        # No need to return edges, nn used hereon
        return nn


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
        m = self.m
        n = self.n
        rselect = self.rselect
        rcount = self.rcount
        q = self.q

        if self.gtype == 'ba':
            args = (nodes, rselect, rcount, n, m)
        elif self.gtype == 'rnd':
            args = (nodes, n, m)
        else:
            args = (nodes, nn, n, m, q)
        # Pass into numba-enabled func, update state attributes
        self.nodes, self.rselect, self.rcount, self.n, self.nn = self._addNode_(*args)
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

    def save(self):
        """
        Saves the graph in a .npy file in directory dependent on graph attributes.
        Returns filepath.
        """
        n = 1
        file_path = f'Data/{self.gtype}/{self.initialType}_initial/N-{self.N}_m-{self.m}_n0-{self.n_0}_{n}.npy'
        while os.path.exists(file_path):
            n += 1
            file_path = f'Data/{self.gtype}/{self.initialType}_initial/N-{self.N}_m-{self.m}_n0-{self.n_0}_{n}.npy'

        with open(file_path, 'wb') as file:
            k, freq = self.bin(self.scale)
            np.save(file, {'Plot' : [k, freq], 'N' : self.N, 'm' : self.m, 'n_0' : self.n_0, 'gtype' : self.gtype, 'initial' : self.initialType})
        return file_path

    def exe(self, save = True):
        """
        Complete execution function for a specified graph.
        """
        print("MAKING INITIAL GRAPH...\n")
        self.initialGraph()
        print("INITIAL GRAPH COMPLETE\n")
        self.growToN()
        if save:
            self.save()
            print("DISTRIBUTION SAVED\n")
        print("GRAPH COMPLETE")
        return None

# ------------------------------------- Plotting ----------------------------------------------
    def plotDegree(self, plot = True):
        """
        Plot degree against frequency using logbin
        """
        k, freq = logbin(self.nodes, scale = self.scale)
        plt.grid()
        plt.ylabel('Frequency')
        plt.xlabel('Degree')
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(k, freq, 'x', label = f'm = {self.m}')
        if plot:
            plt.legend()
            plt.show()
        return [k, freq]
# --------------------------- Retrieve Graph Degrees -----------------------------------
    def getAllDegrees(self):
        """
        Print self.nodes, nparray of N items where node[i] = k(i).
        """
        print("LIST OF NODE DEGREES\n", self.nodes)
        return self.nodes
