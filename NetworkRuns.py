"""
Execution of the tasks required for the networks report using TheGraph
class in the NetworkClass module.

J. J. Window
Imperial College London
Complexity & Networks - Networks Project
"""
# -------------------------- MODULES ---------------------------------
import numpy as np
from NetworkClass import *
import matplotlib.pyplot as plt
import tqdm
from scipy.optimize import curve_fit
# --------------------------- TASKS ------------------------------------

# PHASE 2: PPA
def phase2_distribution(makeGraphs = False, plotpk = True):
    """
    Plots the unbinned and binned log-log degree distributions for all values of
    m.

    PARAMS
    -----------------------------------------------------------------------------------------
    makeGraphs  :   bool, default=False. Indicates whether new graph are to be run,
                    or whether data should be taken from saved graph arrays.
    plotpk      :   bool, default=True. If true, both an unbinned plot and a log
                    binned plot (scale = 1.2) are shown.
    
    RETURNS
    -----------------------------------------------------------------------------------------
    None
    """

    def phase2makeGraphs():
        """
        Function used for generating single instances of a network of fixed size,
        varaible m for m = 3**i, 1 <= i < 6.

        PARAMS
        ----------------------------------------------------------------------------------
        None.

        RETURNS
        ----------------------------------------------------------------------------------
        filepaths   :   list, contains location of saved node degree arrays.
        """
        filepaths = []
        for i in range(1, 6):
            g = 0
            m = 3**i
            # From Poisson distribution
            p_acc = 2*m/999
            # Instantiate graph
            g = TheGraph(m, 1000000, n_0=1000, gtype='ba', initial = 'er', p_acc = p_acc)
            # Run full graph
            filepath = g.exe(multirun = True, bar = False)
            filepaths.append(filepath)
        return filepaths

    def phase2plot(filepaths):
        """
        Plots unbinned, then binned p-k log-log plots for distributions saved in
        the destinations in filepaths array.

        PARAMS
        ---------------------------------------------------------------------------------
        filepaths   :   iterable, should contain valid filepath strings for data to be
                        plotted.
        RETURNS
        ---------------------------------------------------------------------------------
        None.
        """
        # Plot unbinned
        plot_p_k(filepaths)
        # Plot binned
        plot_p_k(filepaths, scale=1.2)
        return None

    if makeGraphs:
        filepaths = phase2makeGraphs()
    else:
        filepaths = []
        for i in range(1, 6):
            i = 1
            filepath = f"Data/multiruns/ba/N-1000000_m-{3**i}_n0-1000_{i}.npy"
            while os.path.exists(filepath):
                i += 1
                filepath = f"Data/multiruns/ba/N-1000000_m-{3**i}_n0-1000_{i}.npy"
            filepaths.append(filepath)
    if plotpk:
        phase2plot(filepaths)
    return None

def multiRun(numruns):
    """
    Repeatedly runs networks for all m, fixed N, multiple times. Degree arrays of each
    network are saved and can be accessed later.

    PARAMS
    --------------------------------------------------------------------------------------
    numruns :   int, number of repeats of each graph to be executed.

    RETURNS
    --------------------------------------------------------------------------------------
    None.
    """
    with tqdm.tqdm(total = numruns, desc = "ALL SIZES ") as bar:
        i = 0
        while i < numruns:
            phase2(True, False, False)
            bar.update()
            i += 1
    return None

def plotAvgAllm(gtype, idx = -1, scale = 1.2):
    """
    Plots the p-k log-log graph using the values averaged across all repeats for each m.
    PARAMS
    ------------------------------------------------------------------------------------
    gtype   :   str, 'ba' 'rnd' or 'rndwlk'. Used to find files to plot.
    idx     :   int, default = -1. Index of calculated average array, if more than one 
                average has been calculated for each m-value. If -1, the most recent 
                average index of m=3 is used as the index for all m. Otherwise, the index
                is used directly.
    scale   :   float, default = 1.2. Log binning scale, passed to logbin function.

    RETURNS
    -------------------------------------------------------------------------------------
    None.
    """
    if idx == -1:
        # Last average calculated - test filepath for m = 3. Assumes each m-value has as many repeats.
        n = 1
        path = f'Data/multiruns/{gtype}/avg/m-{3}_{n}.npy'
        while os.path.exists(path):
            n += 1
            path = f'Data/multiruns/{gtype}/avg/m-{3}_{n}.npy'
        # Last iindex before loop exited - last file which exists.
        idx = n-1
    # Array of filepaths for all m
    filepaths = [f'Data/multiruns/{gtype}/avg/m-{3**i}_{idx}.npy' for i in range(1,6)]
    # Plot graphs
    plot_p_k(filepaths, gtype = gtype, avg = True, scale = scale)
    return None

def getKS(gtype = 'ba'):
    """
    Get K-S statistic for all m-values compared the theoretical distribution.

    PARAMS
    -------------------------------------------------------------------------------------
    gtype   :   str, 'ba' 'rnd' or 'rndwlk'. Type of graph, used to select theoretical 
                function for passing to ks().
    
    RETURNS
    -------------------------------------------------------------------------------------
    testKS  :   dict, contains key-value pairs returned by ks() in NetworkClass.
    """
    filepaths = [f'Data/multiruns/{gtype}/avg/m-{3**i}_2.npy' for i in range(1,6)]
    testKS = ks(filepaths, gtype)
    for i in range(1,6):
        m = 3**i
        print(f'{m}: D = {testKS[m][0]}, n = {testKS[m][1]}')
    return testKS

def largestDegree(m, gtype, makeGraphs = False, numruns = 20, fit=True, theory = False):
    """
    Investigation of k1 for fixed m, varying N. N fixed to vary between 1e5 and 1e6.

    PARAMS
    ---------------------------------------------------------------------------------------
    m           :   int, network parameter, fixed for all networks.
    makeGraphs  :   bool, if true then all network sizes and repeats are run afresh and saved.
    numruns     :   int, default = 20. Number of repeats of each size of graph, over which an
                    average is taken.
    fit         :   bool, default = True. If true then the k1-N graph is plotted with an 
                    optimised fit line.
    theory      :   bool, default = False. If true then the k1-N graph is plotted with the 
                    theoretical relationship.
    """
    def f_theory_ppa(N, m):
        # Theoretical relation for ppa
        return 0.5*(np.sqrt(1+4*m*N*(m+1))-1)

    def f_theory_pra(N, m):
        # Theoretical relation for pra
        return m + np.log(N)/np.log((m+1)/m)

    # Fixed array sizes
    N_arr = [i for i in range(100000, 1000001, 100000)]
    # Execute graph repeats
    if makeGraphs:
        with tqdm.tqdm(desc = "K1: ", total = 10*numruns) as bar:
            for N in N_arr:
                i = 0
                while i < numruns:
                    g = 0
                    g = TheGraph(m, N, gtype=gtype, initial='er', n_0=1000, p_acc = 2*m/999)
                    g.exe(bar=False, k1=True)
                    i += 1
                    bar.update()

    print("AVERAGING...")
    k1 = []
    sig_k1 = []
    with tqdm.tqdm(total=10) as bar:
        for N in N_arr:
            filepaths = [f'Data/multiruns/{gtype}/k1/N-{N}_{i}.npy' for i in range(1, numruns)]
            unpickled = getSavedArray(filepaths)
            degrees = [netwrk[()]['Plot'] for netwrk in unpickled]
            # Calculate mean, std of max degrees
            k1_arr = [max(degree) for degree in degrees]
            k1.append(np.mean(k1_arr))
            sig_k1.append(np.std(k1_arr))
            bar.update()

    # Assign theoretical function based on graph type
    if gtype == 'ba':
        f_theory = f_theory_ppa
    elif gtype == 'rnd':
        f_theory = f_theory_pra

    def f_fit_ppa(N, A, b):
        # Optimisable fit function of power law form
        return A*(N**b)

    def f_fit_pra(N, A, b):
        # Optimisable ln fit function
        return A + np.log(N)/np.log((A+1)/A)
    
    # Assign fit function based on graph type
    if gtype == 'ba':
        f_fit = f_fit_ppa
        p0 = [81, 0.5]
    elif gtype == 'rnd':
        f_fit = f_fit_pra
        p0 = [9,0.1]

    # Setup plot
    font_prop = plotSetup('N', 'Largest Degree')
    # Plot data with errors
    plt.errorbar(N_arr, k1, yerr = sig_k1, fmt = 'x', color = 'red', label = "Measured Data", markersize = 12)
    if theory:
        # Plot theoretical function
        N_th = np.linspace(min(N_arr), max(N_arr), 1000)
        th = [f_theory(_N, m) for _N in N_th]
        plt.plot(N_th, th, '--k', label = "Theory")
        diff = max([(f_theory(_N, m) - k1[i])/f_theory(_N, m) for i, _N in enumerate(N_arr)])
    if fit:
        # Plot optimised fit
        popt, pcov = curve_fit(f_fit, N_arr, k1, p0=p0)
        perr = np.sqrt(np.diag(pcov))
        N_fit = np.linspace(min(N_arr), max(N_arr), 1000)
        fit = [f_fit(_N, popt[0], popt[1]) for _N in N_fit]
        if gtype == 'ba':
            plt.plot(N_fit, fit, '-.k', label = "Power law fit")
        elif gtype == 'rnd':
            plt.plot(N_fit, fit, '-.k', label = "Log(N) fit")
    plt.legend(prop=font_prop)
    plt.show()
    if fit:
        # return fit data
        return k1, [popt[0], perr[0]], [popt[1], perr[1]]
    elif theory:
        print("MAX DIFF: ", diff)
        return k1, diff
    else:
        return k1

# A, b = largestDegree(81, 'ba', makeGraphs=False, numruns = 40)
# print(f"FITTED POWER: {b[0]} +/- {b[1]}")
# print(f"A = {A[0]} +/- {A[1]}")

def makeNetworks(gtype, nruns = 20):
    """
    Make numruns iterations of networks with fixed parameters for p-k investigation.
    N fixed at 1e6, m = 3**i, 1 <= i < 6.
    n_0 = 1000, p_acc = 2m/(n_0-1).

    PARAMS
    ----------------------------------------------------------------------------------
    gtype   :   str, 'ba' 'rnd' or 'rndwlk'. Type of graph to be implemented.
    nruns   :   int, default = 20. Number of repeated graphs to be run for each 
                different m.
    RETURNS
    ----------------------------------------------------------------------------------
    None.
    """
    if gtype == 'rndwlk':
        N = 10000
        n_0 = 100
        q_arr = [0.0, 0.1, 0.5, 0.8, 0.95]
        m = 8
        p_acc = 2*m/(n_0-1)
        for q in q_arr:
            n = 0
            with tqdm.tqdm(desc = f"q = {q}", total = nruns) as bar:
                while n < nruns:
                    g = 0
                    g = TheGraph(m, N, gtype, initial='er', n_0=n_0, p_acc=p_acc, q = q)
                    path = g.exe(multirun = True, bar = False)
                    n += 1
                    bar.update()
    else:
        N = 1000000
        n_0 = 1000
        for i in range(1,6):
            m = 3**i
            p_acc = 2*m/(n_0-1)
            n = 0
            with tqdm.tqdm(desc = f"m = {m}", total = nruns) as bar:
                while n < nruns:
                    g = 0
                    g = TheGraph(m, N, gtype, initial='er', n_0=n_0, p_acc=p_acc)
                    g.exe(multirun = True, bar=False)
                    n += 1
                    bar.update()
    return None

def dataCollapse(gtype, numruns):
    """
    Produce data collapse plot used in report.
    """
    N_arr = [i for i in range(100000, 1000001, 200000)]
    colours = ['red', 'blue', 'green', 'orange', 'yellow']
    k1 = []
    unscaled = []
    filepaths = [f'Data/multiruns/ba/k1/avg/N-{N}_1.npy' for N in N_arr]
    unpickled = getSavedArray(filepaths)
    k_arr = [netwrk[()]['k'] for netwrk in unpickled]
    p_arr = [netwrk[()]['p'] for netwrk in unpickled]

    m = 81
    k1_theory = [k1_theory_ppa(N, m) for N in N_arr]

    k1_arr = [max(k) for k in k_arr]
    font_prop = plotSetup(xlabel = '$k/k_1$', ylabel = '$p/p_{max}$')
    i = 0
    for i, N in enumerate(N_arr):
        if i % 2:
            marker = '.'
        else:
            marker = 'x'
        k_scaled, p_scaled = rescale(k_arr[i], p_arr[i], k1_arr[i], 1)
        kmax = k1_theory[i]
        k_lin = np.linspace(m, kmax, 1000)
        p_th = [theoreticalProb_ppa(m, k) for k in k_lin]
        k_theory_scaled, p_theory_scaled = rescale(k_lin, p_th, kmax, N)
        plt.plot(k_scaled, p_scaled, marker, markersize = 12, color = colours[i], label = f'N = {N}')
    plt.legend(prop = font_prop)
    plt.show()
    return None
