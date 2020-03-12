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
import matplotlib.pyplot as plt

# --------------------Retrieving saved data-------------------------

def plotFile(filepath):
    with open(filepath, 'rb') as file:
        graphDict = np.load(file, allow_pickle=True)
    print(graphDict)
    graph = graphDict[()].get('Plot')
    N = graphDict[()]['N']
    m = graphDict[()]['m']

    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(f'N = {N}')
    plt.plot(graph[0], graph[1], 'x', color = 'black', label = f'm = {m}')
    plt.show()
    return None

filepath = 'Data/ba/er_initial/N-1000_m-3_n0-50_2.npy'

plotFile(filepath)