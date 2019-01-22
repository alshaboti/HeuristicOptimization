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

def pgmpy_test2():
    # example from https://github.com/pgmpy/pgmpy/blob/dev/examples/Learning%20from%20data.ipynb
    # Generating radom data with each variable have 2 states and equal probabilities for each state

    raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
    data = pd.DataFrame(raw_data, columns=['D', 'I', 'G', 'L', 'S'])

    model = BayesianModel([('D', 'G'), ('I', 'G'), ('I', 'S'), ('G', 'L')])

    # Learing CPDs using Maximum Likelihood Estimators
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    for cpd in model.get_cpds():
        print("CPD of {variable}:".format(variable=cpd.variable))
        print(cpd)
def pomegranate_test():
    mydb = np.array([[1,1,1],[1,1,1],[0,1,1]])

    mymodel = BayesianNetwork.from_samples(mydb)

    # print(mymodel.node_count())

    # mymodel.plot()

    print( mymodel.probability([[1,1,1]]) )
    print( mymodel.probability([[None,1,1]]) )
    print( mymodel.predict_proba({}) )

    # print(mymodel.to_json())
# from https://github.com/jmschrei/pomegranate/blob/master/examples/bayesnet_asia.ipynb
def pomegranate_User_pref():
    door_lock = DiscreteDistribution({'True': 0.7, 'False': 0.3})

    thermostate = ConditionalProbabilityTable(
        [['True', 'True', 0.2],
         ['True', 'False', 0.8],
         ['False', 'True', 0.01],
         ['False', 'False', 0.99]], [door_lock])
    clock_alarm = DiscreteDistribution( { 'True' : 0.8, 'False' : 0.2} )

    light = ConditionalProbabilityTable(
        [[ 'True', 'True', 'True', 0.96 ],
         [ 'True', 'True', 'False', 0.04 ],
         [ 'True', 'False', 'True', 0.89 ],
         [ 'True', 'False', 'False', 0.11 ],
         [ 'False', 'True', 'True', 0.96 ],
         [ 'False', 'True', 'False', 0.04 ],
         [ 'False', 'False', 'True', 0.89 ],
         [ 'False', 'False', 'False', 0.11 ]], [door_lock, clock_alarm])

    coffee_maker = ConditionalProbabilityTable(
        [[ 'True', 'True', 0.92 ],
         [ 'True', 'False', 0.08 ],
         [ 'False', 'True', 0.03 ],
         [ 'False', 'False', 0.97 ]], [clock_alarm] )

    window = ConditionalProbabilityTable(
        [[ 'True', 'True', 0.885 ],
         [ 'True', 'False', 0.115 ],
         [ 'False', 'True', 0.04 ],
         [ 'False', 'False', 0.96 ]], [thermostate] )


    s0_door_lock = State(door_lock, name="door_lock")
    s1_clock_alarm = State(clock_alarm, name="clock_alarm")
    s2_light = State(light, name="light")
    s3_coffee_maker = State(coffee_maker, name="coffee_maker")
    s4_thermostate = State(thermostate, name="thermostate")
    s5_window = State(window, name="Window")
    network = BayesianNetwork("User_pref")
    network.add_nodes(s0_door_lock, s1_clock_alarm, s2_light, s3_coffee_maker, s4_thermostate, s5_window)

    network.add_edge(s0_door_lock,s2_light)
    network.add_edge(s0_door_lock,s4_thermostate)
    network.add_edge(s1_clock_alarm,s3_coffee_maker)
    network.add_edge(s1_clock_alarm,s2_light)
    network.add_edge(s4_thermostate, s5_window)
    network.bake()
    return network


network = pomegranate_User_pref()
network.plot()
print(network.probability(['False', 'False', 'False', 'False', 'False', 'False', 'True', 'True']))

#pgmpy_test2()