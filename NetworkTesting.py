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

testBA = TheGraph(2, 1000)
testBA.initialGraph()
testBA.growToN()
testBA.plotDegree()