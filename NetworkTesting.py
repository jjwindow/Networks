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

testBA = TheGraph(3, 1000, n_0=50,
                 initial = 'er', p_acc = 0.8)
testBA.exe()

