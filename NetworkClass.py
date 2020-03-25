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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import tqdm
import os
from collections import Counter
from itertools import combinations, repeat
from math import factorial
import scipy.stats as stat

# ------------------------NETWORK CLASS-----------------------------

class TheGraph:
# ---------------------------------- Initial graphs ------------------------------------------

    @staticmethod
    @numba.njit()
    def _circle_(nodes, rselect, n_0, p_acc):
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
        edges = np.asarray([(0,0) for i in range(n_0)])
        edges[0][0] = n_0                       # First and last edges
        edges[0][1] = 1
        edges[n_0][0] = n_0-1
        edges[n_0][1] = 0
        i = 1
        while i < n_0:                          # Other edges in circle
            edges[i][0] = i-1
            edges[i][1] = i+1
            i += 1

        return nodes, rselect, rcount, edges, n_0

    @staticmethod
    # @numba.njit()
    def _er_init_(nodes, rselect, n_0, p_acc):
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
        attached = [-1 for i in range(m)]   # cannot be zeros since 0 is a node in the graph
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
        attached = [-1 for i in range(m)]   # cannot be zeros since 0 is a node
        isAttached = False
        while _m < m:
            _nextNode = np.random.choice(n)
            attachedArr = [att == _nextNode for att in attached]
            for val in attachedArr:
                # Check if any vals in array are true (any() not usable w/ numba)
                if val:
                    isAttached = True
            while isAttached:
                # repeat until node not already attached
                _nextNode = np.random.choice(n)
                isAttached = False
                attachedArr = [att == _nextNode for att in attached]
                for val in attachedArr:
                    # Check is any vals in array are true (any() not usable w/ numba)
                    if val:
                        isAttached = True
            # new node not attached after loops
            nextNode = _nextNode
            walk = (np.random.random() <= q)
            # Go on a random walk:
            while walk:
                _nextNode = np.random.choice(nn[nextNode])
                # Filler vals in nn[nextNode] are -1, exclude these:
                while _nextNode == -1:
                    # print("2nd attach or -1 loop enter")
                    _nextNode = np.random.choice(nn[nextNode])
                nextNode = _nextNode
                walk = (np.random.random() <= q)
            attached[_m] = nextNode
            nodes[nextNode] += 1
            nn[n][_m] = nextNode
            empty = np.where(nn[nextNode]==-1)
            idx = empty[0][0]               # Index of first empty element of nn[i]
            nn[nextNode][idx] = n
            _m += 1                         # increase m counter
        n += 1                              # one more node in network
        rselect = None                      # Needed for num of returns.
        rcount = None               
        return nodes, rselect, rcount, n, nn


# ---------------------------------- __init__ ------------------------------------------
    def __init__(self, m, N, gtype = 'ba', initial = 'c', n_0 = 100, p_acc = 0.2, q = 0.5):
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
        # # 4mN large enough for initial graph + extra 2*m*N degrees from graph
        # edgenum = int(factorial(n_0)/(2*factorial(n_0-2))) # Max number of edges in ER graph
        # self.edges = np.asarray([(0,0) for i in range(edgenum)]) # empty edge list
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
        # edges = self.edges

        self.nodes, self.rselect, self.rcount, edges, self.n = self._initial_(nodes, rselect, n_0, p_acc)
        if self.gtype == 'rndwlk':
            nn = self.nn
            breakpoint()
            self.nn = self.edgesToNeighbours(edges, nn)
        return None

    @staticmethod
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
        # # Remove empty edges from list
        # edges = [e for e in edges if not all(e)]
        for n1, n2 in edges:
            # First 'empty' elements of nn[n1,2].
            empty1 = np.where(nn[n1] == -1)
            empty2 = np.where(nn[n2] == -1)
            idx1 = empty1[0][0]                     
            idx2 = empty2[0][0]
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
            nn = self.nn
            args = (nodes, nn, n, m, q)
        # Pass into numba-enabled func, update state attributes
        self.nodes, self.rselect, self.rcount, self.n, self.nn = self._addNode_(*args)
        return None

    def growToN(self, bar=True):
        """
        Grows network to final state.
        """
        if bar:
            with tqdm.tqdm(total=self.N, desc = f"NETWORK N: {self.N}", initial = self.n_0) as bar:
                while self.n < self.N:
                    self.addNode()
                    bar.update()
        else:
            while self.n < self.N:
                self.addNode()
        return None

    def save(self, multirun = False, k1 = False, avg=False):
        """
        Saves the graph in a .npy file in directory dependent on graph attributes.
        Returns filepath.
        """
        
        if multirun:
            n = 1
            filepath = f'Data/multiruns/{self.gtype}/N-{self.N}_m-{self.m}_n0-{self.n_0}_{n}.npy'
            while os.path.exists(filepath):
                n += 1
                filepath = f'Data/multiruns/{self.gtype}/N-{self.N}_m-{self.m}_n0-{self.n_0}_{n}.npy'
        elif k1:
            n = 1
            filepath = f'Data/multiruns/{self.gtype}/k1/N-{self.N}_{n}.npy'
            while os.path.exists(filepath):
                n += 1
                filepath = f'Data/multiruns/{self.gtype}/k1/N-{self.N}_{n}.npy'
        else:
            n = 1
            filepath = f'Data/{self.gtype}/{self.initialType}_initial/variable_p/N-{self.N}_m-{self.m}_n0-{self.n_0}_{n}.npy'
            while os.path.exists(filepath):
                n += 1
                filepath = f'Data/{self.gtype}/{self.initialType}_initial/variable_p/N-{self.N}_m-{self.m}_n0-{self.n_0}_{n}.npy'

        with open(filepath, 'wb') as file:
            np.save(file, {'Plot' : self.nodes, 'N' : self.N, 'm' : self.m, 'n_0' : self.n_0, 'gtype' : self.gtype, 'initial' : self.initialType, 'p_acc' : self.p_acc})
        return filepath

    def exe(self, save = True, multirun = False, bar = True, k1 = False):
        """
        Complete execution for a specified graph.
        """
        if bar:
            print("MAKING INITIAL GRAPH...\n")
        self.initialGraph()
        if bar:
            print("INITIAL GRAPH COMPLETE\n")
        self.growToN(bar)
        filepath = None
        if save:
            filepath = self.save(multirun, k1)
            if bar:
                print("DISTRIBUTION SAVED\nFilepath = ", filepath)
        if bar:
            print("GRAPH COMPLETE")
        return filepath

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
# -------------------------------- END OF CLASS ----------------------------------------

######################## FUNCTIONS FOR PLOTTING & ANALYSIS #############################

# ------------------------------- GET SAVED DATA ---------------------------------------
def getSavedArray(filepaths):
    """
    Retrieves the saved .npy files from the locations in 'filepaths' and
    unpickles them. The result is an array of dictionaries containing the 
    attributes of the networks, including the array of node degrees.

    PARAMS
    --------------------------------------------------------------------------
    filepaths   :   iterable, should contain the file paths for the pickled 
                    numpy arrays to be retrieved as binary strings.

    RETURNS
    --------------------------------------------------------------------------
    unpickled   :   list, each element is a dictionary of the format specified
                    in TheGraph.save().
    """
    # Plot distributions
    unpickled = []
    for filepath in filepaths:
        # Unpickle saved array
        with open(filepath, 'rb') as file:
            netwrk = np.load(file, allow_pickle=True)
        unpickled.append(netwrk)
    return unpickled
# ------------------------- THEORETICAL DISTRIBUTIONS ----------------------------------
def theoreticalProb_ppa(m, k):
    """
    Calculates the theoretical probability of finding degree k in a network
    of given m.

    PARAMS
    -----------------------------------------------------------------
    m       :   int, network attribute.
    k       :   float, degree for which probability is to be calcualted

    RETURNS
    -----------------------------------------------------------------
    P_(infinity)(k) for given m [float].
    """
    if type(m) is not int:
        m = m[0]
    if k < m:
        return 0
    else:
        return (2*m*(m+1))/(k*(k+1)*(k+2))
# --------------------------------- PLOTTING -----------------------------------------
def theoreticalPlot(m, kmax, gtype = 'ba'):
    """
    Return plot arrays for the theoretical fit of a probability
    distribution of given m, between kmin and kmax. 
    Range extended to be (0.9*kmin, 1.2*kmax) so the line can be seen 
    clearly on the plot.

    PARAMS
    -----------------------------------------------------------------
    m       :   int, network attribute.
    kmax    :   int, maximum degree in dataset

    RETURNS
    -----------------------------------------------------------------
    k_arr   :   list, range of k values to be plotted on graph.
    p_arr   :   list, p vaues corresponding to the theoretical
                probability distribution on the range of k.
    """
    if gtype == 'ba':
        theoreticalProb = theoreticalProb_ppa
    k_arr = np.linspace(m, 1.5*kmax, 1000)
    p_arr = [theoreticalProb(m, k) for k in k_arr]
    return k_arr, p_arr

def plotSetup(xlabel, ylabel, xscale = 'log', yscale = 'log'):
    """
    Sets up the standard plot format used for the report. Saves repeating setup
    for every type of plot.

    PARAMS
    ----------------------------------------------------------------------------
    xlabel      :   str, the parameter passed to plt.xlabel.
    ylabel      :   str, the parameter passed to plt.ylabel.
    xscale      :   str, parameter passed to plt.xscale.
    yscale      :   str, parameter passed to plt.yscale.

    RETURNS
    ----------------------------------------------------------------------------
    font_prop   :   FrontProperties object, contains the properties of the 
                    Computer Modern Roman Serif font used in the plot. Returned
                    so that it may be used on the other plot elements outside 
                    this function.
    """
    # set tick width
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.size'] = 5
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['ytick.minor.size'] = 5
    mpl.rcParams['ytick.minor.width'] = 1
    # Get font properties - font chosen is Computer Modern Serif Roman (LaTeX font).
    font_path = 'cmunrm.ttf'
    font_prop = font_manager.FontProperties(fname=font_path, size=18)
    # Set up plot
    plt.grid()
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel, fontproperties = font_prop)
    plt.ylabel(ylabel, fontproperties = font_prop)
    # Font has no '-' glyph, so matplotlib serif family used for tick labels.
    plt.xticks(fontfamily = 'serif', fontsize = 14)
    plt.yticks(fontfamily = 'serif', fontsize = 14)
    return font_prop

def plot_p_k(filepaths, avg = False, scale = 1., fit = True):
    """
    Plotting function for log(p) vs log(k) graph.

    PARAMS
    -----------------------------------------------------------------------
    filepaths   :   iterable, contains list of filepaths of saved numpy arrays
                    to be plotted on the graph.
    scale       :   float, scale used in log binning. Default = 1, corresponds 
                    to no binning. 1.2 suggested for reasonable binning.
    fit         :   bool, default True. Determines whether to plot the theoretical
                    distribution for each value of m.

    RETURNS
    ------------------------------------------------------------------------
    None.
    """
    # Define colours
    colours = ['red', 'blue', 'green', 'orange', 'yellow']
    # Retrieves pickled network data
    unpickled = getSavedArray(filepaths)
    if avg:
        plots = [[netwrk[()]['k'], netwrk[()]['p'], netwrk[()]['sig_p']] for netwrk in unpickled]
        m_arr = [3**i for i in range(1, 6)]
    else:
        degrees = [netwrk[()]['Plot'] for netwrk in unpickled]
        m_arr = [netwrk[()]['m'] for netwrk in unpickled]
        plots = [logbin(degree, scale) for degree in degrees]

    # Sets up plot, returns font properties
    font_prop = plotSetup(xlabel = "Degree k", ylabel = "Log(p)")
    # Plot distributions
    for i, plot in enumerate(plots):
        m = m_arr[i]
        k = plot[0]
        freq = plot[1]
        sig_p = plot[2]
        if m==3 and avg:
            k = k[1:]
            freq = freq[1:]
            sig_p = sig_p[1:]
        if scale == 1.:
            # Plot dots for unbinned data - makes bins easier to see at higher degrees.
            plt.plot(k, freq, '.', color = colours[i], label = f'm = {m}')
        else:
            # For binned data, plot crosses to make points more obvious.
            if avg:
                # Averaged plots need errorbars
                plt.errorbar(k, freq, yerr=sig_p, fmt='x', markersize = '8', color = colours[i], label = f'm={m}')
            else:
                plt.plot(k, freq, 'x', markersize = 8, color = colours[i], label = f'm = {m}')
        if fit:
            # Plot theoretical distribution
            kmin = min(k)
            kmax = max(k)
            k_arr, p_arr = theoreticalPlot(m, kmax)
            p_arr = [p for p in p_arr if p > 1e-7]
            k_arr = k_arr[:len(p_arr)]
            plt.plot(k_arr, p_arr, '-', color = colours[i])
    plt.legend(prop = font_prop)
    plt.show()
    return None
# ------------------------------------- ANALYSIS ----------------------------------------------
def multiAvg(m, gtype, nruns, degrees, scale, k1=False, N=1e6):
    """
    Average the log-binned probability distributions for multiple runs
    of a network of fixed N, m. Saves the log-binned average distribution in
    destpath.

    PARAMS
    -------------------------------------------------------------------------------
    m       :   int, number of edges added to each new node in network.
    gtype   :   str, 'ba', 'rnd' or 'rndwlk'. Refers to the algorithm used
                to grow the network in full; either pure preferential attachment,
                pure random attachment, or random walk respectively.
    nruns   :   int, number of graph instances to average results over.
    degrees :   iterable, array of node degrees of a fully grown network.
    scale   :   float, parameter passed to logbin function. 1 corresponds to no 
                binning. 1.2 is typical.
    k1      :   bool, default = False. Whether the data is going to be used to 
                find k1 of a given network. If true, N is saved in the disctionary 
                instead of m.
    N       :   int, default = 1e6. Number of nodes in the network. Does not need
                to be specified unless k1=True.
    
    RETURNS
    --------------------------------------------------------------------------------
    destpath:   str, file path for the saved .npy file containing the averaged
                distribution.

    """
    # Define destination filepath
    i = 1
    if k1:
        # Different filepath needed for k1 average
        destpath = f'Data/multiruns/{gtype}/k1/avg/N-{N}_{i}.npy'
        while os.path.exists(destpath):
            i += 1
            destpath = f'Data/multiruns/{gtype}/k1/avg/N-{N}_{i}.npy'
    else:    
        destpath = f'Data/multiruns/{gtype}/avg/m-{m}_{i}.npy'
        while os.path.exists(destpath):
            i += 1
            destpath = f'Data/multiruns/{gtype}/avg/m-{m}_{i}.npy'
    
    # Logbin degree arrays
    raw_k_p = [logbin(degree, scale = scale) for degree in degrees]
    # Separate into k, p
    raw_k = [binned[0] for binned in raw_k_p]
    raw_p = [binned[1] for binned in raw_k_p]
    # Calculate averages for each bin
    k = [np.mean(k_arr) for k_arr in zip(*raw_k)]
    p = [np.mean(p_arr) for p_arr in zip(*raw_p)]
    # Calculate uncertainty
    sig_p = [np.std(p_arr) for p_arr in zip(*raw_p)]
    # Save averages
    if k1:
        np.save(destpath, {'k' : k, 'p' : p, 'sig_p' : sig_p, 'nruns' : nruns, 'N' : N})
    else:
        np.save(destpath, {'k' : k, 'p' : p, 'sig_p' : sig_p, 'nruns' : nruns, 'm' : m})
    return destpath

def avgAllm(nruns, gtype = 'ba'):
    """
    Average the log-binned degree distributions for all runs of each
    different m, fixed N. 

    PARAMS
    --------------------------------------------------------------------------------------
    nruns   :   int, number of runs over which to take average. Passed into multiAvg().
    gtype   :   str, type of graph grown from 'ba', 'rnd', 'rndwlk'.

    RETURNS
    --------------------------------------------------------------------------------------
    avgpaths:   list, contains filepaths of the saved .npy files for average log-binned 
                distributions of each value of m.
    """
    avgpaths = []
    # m = [3**1, 3**2,...,3**5]
    for i in range(1,6):
        m = 3**i
        # Define filepaths
        filepaths = [f"Data/multiruns/{gtype}/N-1000000_m-{m}_n0-1000_{j}.npy" for j in range(1, nruns)]
        # Get array of node degrees
        unpickled = getSavedArray(filepaths)
        degrees = [netwrk[()]['Plot'] for netwrk in unpickled]
        # Get average for each m across nruns networks, return filepath of average result.
        avgpaths.append(multiAvg(m, gtype, nruns, degrees, 1.2))
    return avgpaths

def ks(filepaths, gtype = 'ba'):
    """
    Implementation of the Kolmogorov-Smirnov test.

    PARAMS
    ----------------------------------------------------------------------------------------
    filepaths   :   iterable, contains filepaths of the data to be tested as strings.
    gtype       :   str, graph type, choice from 'ba', 'rnd', or 'rndwlk'.

    RETURNS
    ----------------------------------------------------------------------------------------
    ks          :   dict, contains key-value pairs as m : [D, n] for each m value of 
                    distributions passed into the function. D [float] is the K-S statistic, defined
                    as the supremum of vertical differences between the empirical 
                    distribution function and the cumulative distribution function of the 
                    expected deistribution. n [int] is the number of data points in the array - 
                    needed when looking up critical value in table.

    """
    # Get pickled arrays
    unpickled = getSavedArray(filepaths)
    m_arr = [netwrk[()]['m'] for netwrk in unpickled]
    k_arr = [netwrk[()]['k'] for netwrk in unpickled]
    p_arr = [netwrk[()]['p'] for netwrk in unpickled]
    # Select theoretical comparison based on graph type
    if gtype == 'ba':
        f_theory = theoreticalProb_ppa

    ks = {}
    for m, k, p in zip(m_arr, k_arr, p_arr):
        # Truncate values k<m.
        new_k = [_k for _k in k if _k >= m]
        new_p = [_p for _k, _p in zip(k, p) if _k >= m]
        p_theory = [f_theory(m, _k) for _k in new_k]
        n = len(new_p)
        # Calculate theoretical cdf
        cdf = [sum(p_theory[:i]) for i in range(n)]
        # Observed edf
        edf = [sum(new_p[:i])/n for i in range(n)]
        diff = [abs(c-e) for c, e in zip(cdf, edf)]
        # K-S statistic, by definition.
        D = max(diff)
        ks[m] =  [D, n]
    return ks
