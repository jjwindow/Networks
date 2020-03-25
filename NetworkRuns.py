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
def phase2_distribution(makeGraphs = False, plotpk = True:
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

def plotAvgAllm(gtype = 'ba', idx = -1):
    """
    Plots the p-k log-log graph using the values averaged across all repeats for each m.
    PARAMS
    ------------------------------------------------------------------------------------
    gtype   :   str, 'ba' 'rnd' or 'rndwlk'. Used to find files to plot.
    idx     :   int, default = -1. Index of calculated average array, if more than one 
                average has been calculated for each m-value. If -1, the most recent 
                average index of m=3 is used as the index for all m. Otherwise, the index
                is used directly.

    RETURNS
    -------------------------------------------------------------------------------------
    None.
    """
    if idx == -1:
        # Last average calculated - test filepath for m = 3. Assumes each m-value has as many repeats.
        n = 0
        path = f'Data/multiruns/{gtype}/avg/m-{3}_{n}.npy'
        while os.path.exists(path):
            n += 1
            path = f'Data/multiruns/{gtype}/avg/m-{3}_{n}.npy'
        # Last iindex before loop exited - last file which exists.
        idx = n-1
    # Array of filepaths for all m
    filepaths = [f'Data/multiruns/{gtype}/avg/m-{3**i}_{idx}.npy' for i in range(1,6)]
    # Plot graphs
    plot_p_k(filepaths, avg = True, scale = 2)
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
    filepaths = [f'Data/multiruns/{gtype}/avg/m-{3**i}_3.npy' for i in range(1,6)]
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

    def f_theory_ppa(N, m):
        # Theoretical relation for ppa
        return 0.5*(np.sqrt(1+4*m*N*(m+1))-1)

    def f_theory_pra(N, m):
        # Theoretical relation for pra
        return m + np.ln(N)/np.ln((m+1)/m)

    # Assign f_theory based on graph type
    if gtype == 'ba':
        f_theory = f_theory_ppa
    elif gtype == 'rnd':
        f_theory = f_theory_pra

    def f_fit_ppa(N, A, b):
        # Optimisable fit function of power law form
        return A*(N**b)

    def f_fit_pra(N, A, b):
        # Optimisable ln fit function
        return A - b*np.ln(N)
    
    # Assign fit function based on graph type
    if gtype == 'ba':
        f_fit = f_fit_ppa
        p0 = [81, 0.5]
    elif gtype == 'rnd':
        f_fit = f_fit_pra
        p0 = [1,1]

    # Setup plot
    font_prop = plotSetup('N', 'Degree k')
    # Plot data with errors
    plt.errorbar(N_arr, k1, yerr = sig_k1, fmt = 'x', color = 'red', label = "Measured Data", markersize = 12)
    if theory:
        # Plot theoretical function
        N_th = np.linspace(min(N_arr), max(N_arr), 1000)
        th = [f_theory(_N, m) for _N in N_th]
        plt.plot(N_th, th, '--k', label = "Theory")
    if fit:
        # Plot optimised fit
        popt, pcov = curve_fit(f_fit, N_arr, k1, p0=p0)
        perr = np.sqrt(np.diag(pcov))
        N_fit = np.linspace(min(N_arr), max(N_arr), 1000)
        fit = [f_fit(_N, popt[0], popt[1]) for _N in N_fit]
        plt.plot(N_fit, fit, '-.k', label = "Power law fit")
    plt.legend(prop=font_prop)
    plt.show()
    if fit:
        # return fit data
        return [popt[0], perr[0]], [popt[1], perr[1]] 
    else:
        return None

A, b = largestDegree(81, 'ba', makeGraphs=False, numruns = 40)
print(f"FITTED POWER: {b[0]} +/- {b[1]}")
print(f"A = {A[0]} +/- {A[1]}")

