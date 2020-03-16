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

# PHASE 2: PPA
def phase2(makeGraphs = False):
    def phase2makeGraphs():
        filepaths = []
        for i in range(5):
            m = 3**i
            g = TheGraph(m, 1000000, n_0=1000, gtype='ba', initial = 'er', p_acc = 0.1)
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
        for i in range(5):
            filepaths.append(f"Data/ba/er_initial/N-1000000_m-{3**i}_n0-1000_1.npy")

    phase2plot(filepaths)
    return None

phase2()