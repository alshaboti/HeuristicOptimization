#!/usr/bin/python3
import numpy as np
import  random
import  itertools
from pomegranate import BayesianNetwork
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, State
from RandomDAG import RandomDAG

randint = random.randint
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


# from https://github.com/jmschrei/pomegranate/blob/master/examples/bayesnet_asia.ipynb
class User_model:
    def __init__(self, nodes, n_edges, devices, func_alter_devices, is_gen_task):

        self.nodes = nodes
        self.n_edges = n_edges
        self.devices = devices 
        self.is_gen_task = is_gen_task
        self.func_alter_devices = func_alter_devices

        self.task_dict = {}
       
        # BN
        self.BN_node_orders = []
        self.network = self.get_BN()
        self.network.bake()
        
        self.task_fucs = self.task_dict.keys()

    def get_score(self, cand_list):
        can_dev = self.build_BN_query(cand_list) 
        # try:
        return self.network.probability(can_dev),
        # except KeyError as ke:
            
        #     print("ERROR!!!!!!!")
        #     print("TASK: ", self.task_fucs)
        #     print("cand: ", cand_list)
        #     print("dev:  ", can_dev)

        #     print("dist:", self.npd)
        #     #print(self.network.states)
        #     print("-----------------")
        #     print(ke.value())
            
        

    def build_BN_query(self, cand_list):
        # first cand_list for first func in task
        can_dev = [None for f in self.nodes]
        
        # the order of nodes and cand_list should be same
        for f,d in zip(self.task_fucs, cand_list):             
            f_idx = self.BN_node_orders.index(f)          
            can_dev[f_idx] = d
     
        return can_dev

    def get_nodes_prob_dist(self, node_without_parents, child_parent):
        node_prob_dict = {}

        for node in node_without_parents:
            n_alters = len(self.func_alter_devices[node])
            p = np.random.random(n_alters)
            p /= p.sum()
            # now the sum of p is 1

            dist = {}
            # node not in the user preference nodes
            # give random probability for all
            if node not in self.task_dict.keys():
                # randomly map p to alter devices
                dist = dict( zip(self.func_alter_devices[node] , p) )

            else:  # set max prob to the perfered alter
                pref_alter = self.task_dict[node]
                dist[pref_alter] = np.amax(p)
                p = list(p)
                p.remove(dist[pref_alter])
                # remove best_alter from alter and set the rest of the prob to them
                alt_list = list(self.func_alter_devices[node])
                alt_list.remove(pref_alter)

                #np.delete(p, np.amax(p))
                dist.update(dict(zip(alt_list, p)))

            # save node with its prob
            node_prob_dict[node] = dist

        #print("Root nodes: ", node_prob_dict)

        # these nodes have parents, generate CPT for them
        for node, parent_lst in child_parent.items():
            # parents + this node condProbTable
            condProbTable = self.getCondProbTable(node, parent_lst)
            # save node with its prob
            node_prob_dict[node] = condProbTable
           # print("child node: ", node, " table:", condProbTable)
        
        return node_prob_dict

    # n_types: number devices types (e.g. door_lock, camera, etc)
    # n_alter: number of devices from same type
    def get_BN(self):

        # 1 Build BN DAG structure
        rand_dag = RandomDAG(self.nodes, self.n_edges)
        DAG, child_parent     = rand_dag.random_dag()
        
        # for a, bs in DAG.edge.items():
        #     for b in bs.keys():
        #         print(a, "->", b)
        ################################################
        # select a random nodes and vlaue as user preference
        if self.is_gen_task:
            selected_tasks = rand_dag.dag_longest_path(DAG)
            for f in selected_tasks:
                func_devices = self.func_alter_devices[f]
                self.task_dict[ f ] = random.choice(func_devices)
        
        ################################################        
        node_without_parents = [e for e in self.nodes if e not in child_parent.keys()]

        # 2 Build BN probability model
        # 2.1 get probabilityDist or conditional prob table
        # bais the prob to task_dict choices
        node_prob_dict = self.get_nodes_prob_dist(node_without_parents, child_parent)
        self.npd = node_prob_dict
        # 2.2 Create nodes linked to its parent, parent should be processed first.
        # all node state saved to be added to the BN later
        nodes_state = {}
        # all node dist or CPT saved to link child to parents when building child CPT
        nodes_dist = {}

        # start with root nodes (don't have parents then link child to them)
        # list the list to copy it, otherwise it will point to the self.nodes 
        remaining_nodes_list = list(self.nodes)

        for node in node_without_parents:
            prob_dist = node_prob_dict[node]
            node_dist = DiscreteDistribution(prob_dist)
            nodes_dist[node] = node_dist
            nodes_state[node] = State(node_dist, name=node)
            # remove from nodes_list
            remaining_nodes_list.remove(node)

        # rest of the node should have parents
        while len(remaining_nodes_list) > 0:
            for node, parent_lst in child_parent.items():
                # if node's parents already created then it can be created now
                if set(parent_lst).issubset(nodes_state.keys()) and \
                    node in remaining_nodes_list:

                    node_dist = ConditionalProbabilityTable(node_prob_dict[node], \
                                    [nodes_dist[i] for i in parent_lst])
                    
                    nodes_dist[node] = node_dist
                    nodes_state[node] = State(node_dist, name=node)
                    # remove from the node_list
                    remaining_nodes_list.remove(node)
        
        # 3 Create BN and add the nodes_state
        self.network = BayesianNetwork("User_pref")
        for node, state in nodes_state.items():
            self.network.add_node(state)
            #print("node ", node, " is added!")
            self.BN_node_orders.append(node)

        # 4 Link nodes with edges using nodes_state and DAG.edge
        for a, bs in DAG.edge.items():
            for b in bs.keys():
                self.network.add_edge(nodes_state[a], nodes_state[b])
        #       print("Network has ", self.network.node_count() , " nodes and ", self.network.edge_count(), " edges")
        return self.network

    def get_permutation_groups(self, parent_node_lst):
        alter_dev = []
        for n in parent_node_lst:
            alter_dev.append( self.func_alter_devices[n] )
            # list(range(n_att))

        permutation = list(dict(zip(parent_node_lst, x)) for x in  itertools.product(*alter_dev) )

       # Gruop the permutation of node alter node
        n_func_dev = len(self.func_alter_devices[parent_node_lst[-1]])
        n_prob_groups = int(len(permutation) / n_func_dev)
        perm_groups = [[] for i in range(n_prob_groups)]
        c = 0
        for perm in permutation:
            # add to the begining
            perm_groups[c // n_func_dev].append(perm)
            c += 1

        return perm_groups

    def getCondProbTable(self, node, parent_lst):
        parent_node_lst = []
        parent_node_lst.extend(parent_lst)
        parent_node_lst.append(node)
        perm_groups_prob = self.get_permutation_groups(parent_node_lst)
        
        #print(perm_groups_prob)
        condProbTable = []

        n_func_dev = len(self.func_alter_devices[ node ])

        maxp_for_best_alter = 1.7 / n_func_dev
        if maxp_for_best_alter < 0.2:
            maxp_for_best_alter = 0.2 

        if self.is_gen_task:
            # check if this child_parent or indp node are in the user prefered devices
            intersect_dict = {k: v for k, v in self.task_dict.items() \
                if k in parent_node_lst}
            #print("intersect_dict:", intersect_dict)
        

        # for each permutation generate prob such that the sum of each node CDP is = 1
        for perm_group_prob in perm_groups_prob:
            if self.is_gen_task and len(intersect_dict) > 1 and \
                    [True for j in range(n_func_dev) \
                        if intersect_dict.items() <= perm_group_prob[j].items()]:

                # print(intersect_dict)
                # generate p such that one value is
                rem_alt = n_func_dev - 1
                rest_prob = np.random.random(rem_alt)
                rest_prob /= sum(rest_prob)
                rest_prob *= (1 - maxp_for_best_alter)

                # the sum of maxp_for_best_alter and rest_prob = 1
                for j in range(n_func_dev):
                    # alter_idx = i * n_att + j
                    condProbRow = list(perm_group_prob[j].values())
                    if intersect_dict.items() <= perm_group_prob[j].items():
                       # print("Best candidate ", perm_group_prob[j], " prob:", maxp_for_best_alter)
                        condProbRow.append(maxp_for_best_alter)
                    else:
                        rem_alt -= 1
                        #print("NOT  candidate ", perm_group_prob[j], " prob:", rest_prob[rem_alt])
                        condProbRow.append(rest_prob[rem_alt])
                   
                    condProbTable.append(condProbRow)

            else:
                # to gurantee best alter, no others should have prob> maxp
                a = np.random.random(n_func_dev)
                a /= a.sum()                
                while self.is_gen_task and np.amax(a) >= maxp_for_best_alter:
                    a = np.random.random(n_func_dev)
                    a /= a.sum()

                # to make sum of alter prob = 1
                for j in range(n_func_dev):
                    condProbRow = list(perm_group_prob[j].values())
                    condProbRow.append(a[j])
                    #print(condProbRow)
                    condProbTable.append(condProbRow)
        return condProbTable
        

