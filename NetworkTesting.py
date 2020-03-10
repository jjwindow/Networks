"""
Testing implementation of Barabasi-Albert network (preferential attachment)

J. J. Window
Imperial College London
Complexity & Networks - Networks Project
"""
# --------------------------MODULES---------------------------------
import numpy as np
from NetworkClass import *

# --------------------------TESTING---------------------------------

testBA = TheGraph(3, 100000, n_0=500
, initial = 'er', p_acc = 0.8)
testBA.exe()
testBA.plotDegree()