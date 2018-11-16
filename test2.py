'''
This module contains some example uses of the PGM library. 
It is intended to be viewed as sample code, but every entry may be run. 
Simply untoggle the print statement of an entry and run the module to see 
that entry in action.
'''
import json
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.lgbayesiannetwork import LGBayesianNetwork
from libpgm.pgmlearner import PGMLearner


# (9) -------------------------------------------------------------------------
# Learn the structure of a discrete Bayesian network, given only data:
nd = NodeData()
skel = GraphSkeleton()

bn = DiscreteBayesianNetwork(skel,nd)

# say I have some data
data = bn.randomsample(2000)

# instantiate my learner 
learner = PGMLearner()

# estimate parameters
result = learner.discrete_constraint_estimatestruct(data)

# output - toggle comment to see
print (json.dumps(result.E, indent=2))



# (11) ----------------------------------------------------------------------
# Learn a structure of a linear Gaussian Bayesian network, given only data
lgbn = LGBayesianNetwork()
# say I have some data
data = lgbn.randomsample(8000)

# instantiate my learner 
learner = PGMLearner()

# estimate parameters
result = learner.lg_constraint_estimatestruct(data)

# output - toggle comment to see
#print json.dumps(result.E, indent=2)

# (12) -----------------------------------------------------------------------
# Learn entire Bayesian networks

# say I have some data
data = lgbn.randomsample(8000)

# instantiate my learner 
learner = PGMLearner()

# estimate parameters
result = learner.lg_estimatebn(data)

# output - toggle comment to see
#print json.dumps(result.E, indent=2)
#print json.dumps(result.Vdata, indent=2)

# say I have some data
data = bn.randomsample(2000)

# instantiate my learner 
learner = PGMLearner()

# estimate parameters
result = learner.discrete_estimatebn(data)

# output - toggle comment to see
#print json.dumps(result.E, indent=2)
#print json.dumps(result.Vdata, indent=2)
