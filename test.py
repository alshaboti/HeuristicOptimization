from pomegranate import BayesianNetwork
from pomegranate import *
import numpy as np
# Defining the Bayesian Model

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import numpy as np
import pandas as pd

# https://github.com/pgmpy/pgmpy
# another example: https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb

def pgmpy_test():

    raw_data = np.array([0] * 30 + [1] * 70)  # Representing heads by 0 and tails by 1
    data = pd.DataFrame(raw_data, columns=['coin'])
    print(data)
    model = BayesianModel()
    model.add_node('coin')

    # Fitting the data to the model using Maximum Likelihood Estimator
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    print(model.get_cpds('coin'))

def pomegranate_test():
    mydb = np.array([[1,1,1],[1,1,1],[0,1,1]])

    mymodel = BayesianNetwork.from_samples(mydb)

    # print(mymodel.node_count())

    # mymodel.plot()

    print( mymodel.probability([[1,1,1]]) )
    print( mymodel.probability([[None,1,1]]) )
    print( mymodel.predict_proba({}) )

    # print(mymodel.to_json())
