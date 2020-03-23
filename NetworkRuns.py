"""
Execution of the tasks required for the networks report using TheGraph
class in the NetworkClass module.

J. J. Window
Imperial College London
Complexity & Networks - Networks Project
"""
# --------------------------MODULES---------------------------------
import numpy as np
from NetworkClass import *
import matplotlib.pyplot as plt
import tqdm

# PHASE 2: PPA
def phase2(makeGraphs = False, plotpk = True, residuals = True):
    def phase2makeGraphs():
        filepaths = []
        for i in range(1, 6):
            m = 3**i
            p_acc = m/1000
            g = TheGraph(m, 1000000, n_0=1000, gtype='ba', initial = 'er', p_acc = p_acc)
            filepath = g.exe()
            filepaths.append(filepath)
        return filepaths

    def phase2plot(filepaths):
        plot_p_k(filepaths)
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
    if residuals:
        plotResiduals(filepaths)
    return None

# phase2(makeGraphs=True, residuals = False)
def exeMultiple(mstart = 1, rangestart = 0):
    # Repeat for m = 3,9,27,81,243
    for i in range(mstart, 6):
        m = 3**i
        p_acc = m/999
        # Repeat each m 50 times
        if i != mstart:
            rangestart = 0

        with tqdm.tqdm(total=50, desc = f"NETWORK m: {m}", initial = rangestart) as bar:
            for j in range(rangestart, 50):
                g = TheGraph(m, 1000000, n_0 = 1000, gtype = 'ba', initial = 'er', p_acc = p_acc)
                # Save in multiruns folder
                g.exe(multirun=True, bar=False)
                bar.update()
    return None

# exeMultiple(5, 35)
m = 3**5
p_acc = m/999
with tqdm.tqdm(total = 15):
    for i in range(15):
        g = TheGraph(m, 1000000, n_0 = 1000, gtype = 'ba', initial = 'er', p_acc = p_acc)
        # Save in multiruns folder
        g.exe(multirun=True, bar=False)
        bar.update()


