from pomegranate import BayesianNetwork
from pomegranate import *
import numpy as np

mydb = np.array([[1,1,1],[1,1,1],[0,1,1]])

mymodel = BayesianNetwork.from_samples(mydb)

# print(mymodel.node_count())

# mymodel.plot()

print( mymodel.probability([[1,1,1]]) )
print( mymodel.probability([[None,1,1]]) )
print( mymodel.predict_proba({}) )

# print(mymodel.to_json())
