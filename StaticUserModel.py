#!/usr/bin/env python

from pomegranate import BayesianNetwork
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, State


# https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4b_Bayesian_Network_Structure_Learning.ipynb
# https://github.com/pgmpy/pgmpy
# another example: https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb

# from https://github.com/jmschrei/pomegranate/blob/master/examples/bayesnet_asia.ipynb

class Static_User_model:
    def __init__(self):
        self.network = self.get_bayesnet()
        self.devices = ['d1','d2','a1','a2','l1','l2','c1','c2',]

    def get_bayesnet(self):
        door_lock = DiscreteDistribution({'d1': 0.7, 'd2': 0.3})

        clock_alarm = DiscreteDistribution( { 'a1' : 0.8, 'a2' : 0.2} )

        light = ConditionalProbabilityTable(
            [[ 'd1', 'a1', 'l1', 0.96 ],
             ['d1', 'a1', 'l2', 0.04 ],
             [ 'd1', 'a2', 'l1', 0.89 ],
             [ 'd1', 'a2', 'l2', 0.11 ],
             [ 'd2', 'a1', 'l1', 0.96 ],
             [ 'd2', 'a1', 'l2', 0.04 ],
             [ 'd2', 'a2', 'l1', 0.89 ],
             [ 'd2', 'a2', 'l2', 0.11 ]], [door_lock, clock_alarm])



        coffee_maker = ConditionalProbabilityTable(
            [[ 'a1', 'c1', 0.92 ],
             [ 'a1', 'c2', 0.08 ],
             [ 'a2', 'c1', 0.03 ],
             [ 'a2', 'c2', 0.97 ]], [clock_alarm] )

        s_door_lock = State(door_lock, name="door_lock")
        s_clock_alarm = State(clock_alarm, name="clock_alarm")
        s_light = State(light, name="light")
        s_coffee_maker = State(coffee_maker, name="coffee_maker")
        network = BayesianNetwork("User_pref")
        network.add_nodes(s_door_lock, s_clock_alarm, s_light, s_coffee_maker)

        network.add_edge(s_door_lock,s_light)
        network.add_edge(s_clock_alarm,s_coffee_maker)
        network.add_edge(s_clock_alarm,s_light)
        network.bake()
        return network

    def get_score(self, cand_list):
        #print(" get_Score: ",cand_list)
        can_dev = []
        for i in range(8): #to do make it general
            if i%2 == 0:
                if i in cand_list:
                    can_dev.append(self.devices[i])
                elif i+1 in cand_list:
                    can_dev.append(self.devices[i+1])
                else:
                    can_dev.append(None)
        #print(can_dev)
        return self.network.probability(can_dev),can_dev
