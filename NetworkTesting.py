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

testBA = BA_net(2, 10000)
testBA.initialGraph_circle(5)
testBA.growToN()
testBA.plotDegree()