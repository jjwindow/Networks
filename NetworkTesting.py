"""
Testing implementation of Barabasi-Albert network (preferential attachment)

J. J. Window
Imperial College London
Complexity & Networks - Networks Project
"""
# --------------------------MODULES---------------------------------
import numpy as np
from NetworkClass import *
import matplotlib.pyplot as plt

# --------------------------TESTING---------------------------------

# testBA = TheGraph(3, 1000, n_0=50,
#                  initial = 'er', p_acc = 0.8)
# testBA.exe()

# testBA = TheGraph(6, 100000, n_0=500, gtype = 'ba',
#                  initial = 'er', p_acc = 0.8)
# testBA.exe()

# for i in range(10):
#     n_0 = 50 + i*25
#     g = TheGraph(5, 100000, n_0 = n_0, gtype='ba', initial = 'c', p_acc = 0.8)
#     print(f"Initial size: {n_0}\n")
#     g.exe()

g = TheGraph(5, 10000, n_0=100, gtype='ba', initial = 'er', p_acc = 0.1)
# g.initialGraph()
# g.plotDegree()
# g.growToN()
# g.plotDegree()
filepath = [g.exe()]
print("FP: ", filepath)
plot_p_k(filepath)
