#!/usr/bin/python3
import numpy as np
import  random
import  itertools
from pomegranate import BayesianNetwork
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, State
from RandomDAG import RandomDAG
# from pprint import pprint

randint = random.randint
# https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4b_Bayesian_Network_Structure_Learning.ipynb
# https://github.com/pgmpy/pgmpy
# another example: https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb

# from https://github.com/jmschrei/pomegranate/blob/master/examples/bayesnet_asia.ipynb

      
def get_dev_funcs(n_functions, \
    min_dev_caps, n_alter_dev_per_func):
    # functions will have fixed number of dev alter
    # however, devices will have cap >= min_dev_caps

    # lots of devices with min_cap
    max_dev_n = int(n_alter_dev_per_func/min_dev_caps * n_functions) 
    # less devices with cap = alter_devices
    min_dev_n = n_functions 

    n_devices = randint(min_dev_n,max_dev_n) 
   
    devices_cap = { "d" + str(i) : [] for i in range(n_devices) }
    functions = [ "f" + str(i) for i in range(n_functions) ]
    func_alter_devices = { f: [] for f in functions} 

    for f in functions:
        n_alt = 0
        while n_alt< n_alter_dev_per_func:            
            # pick rand d not already assigned to the func            
            d = random.choice(list(set(devices_cap.keys()) - set (func_alter_devices[f])))
            func_alter_devices[f].append(d)         
            n_alt += 1

    dev_no_cap = ["d" + str(i) for i in range(n_devices) ]
    for f, d_alt in func_alter_devices.items():
        for d in d_alt:
            devices_cap[d].append(f)
            if d in dev_no_cap:
                dev_no_cap.remove(d)

    for d in dev_no_cap:
        del devices_cap[d]

    return devices_cap, functions, func_alter_devices


# from https://github.com/jmschrei/pomegranate/blob/master/examples/bayesnet_asia.ipynb
class User_model:
    def __init__(self, is_gen_task, n_alter_dev_per_func):
        self.is_gen_task = is_gen_task
        self.n_alter_dev_per_func = n_alter_dev_per_func
        self.task_dict = {}       
        # BN
        self.BN_node_orders = []
        self.devices = None
        self.nodes = None
        self.func_alter_devices = None

    def build_model(self,req_task_len):    
        # 1,2 are based on montcarlo experimnet for tasklen(2 to 10)
        # which return a DAG with less than 100 triels.
        #1-req_task_len is 20% of total fucntions.        
        n_nodes = 5 * req_task_len
        #2- edges three times the task len        
        n_edges = req_task_len * 3 
        min_dev_caps = 2 

        # number of funcitons for each device
        self.devices, self.nodes, self.func_alter_devices = get_dev_funcs(n_nodes, \
                min_dev_caps, self.n_alter_dev_per_func )
        # pprint(self.func_alter_devices,width=1 )

        rand_dag = RandomDAG(self.nodes, n_edges)
        
        DAG, child_parent = rand_dag.get_custom_DAG(req_task_len)
        # print("Child_parents returns by custom DAG: ")
        # pprint(child_parent, width=1)
        
        for f in rand_dag.dag_longest_path(DAG):
            self.task_dict[f] = ''

        # check if we get the task length the at we want            
        for f in self.task_dict.keys():
            func_devices = self.func_alter_devices[f]
            self.task_dict[ f ] = random.choice(func_devices)
        self.task_fucs = self.task_dict.keys()
        # print(self.task_dict)

        self.network = self.get_BN(DAG, child_parent)    
        self.network.bake()       
        
 




    def get_score(self, cand_list):
        can_dev = self.build_BN_query(cand_list) 
        # try:
        return self.network.probability(can_dev),
        
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
            dist = {}

            # node not in the user preference nodes
            # give random probability for all
            if node not in self.task_dict.keys():
                p = np.random.random(n_alters)
                p /= p.sum()
                # now the sum of p is 1
                # randomly map p to alter devices
                dist = dict( zip(self.func_alter_devices[node] , p) )

            else:  # set max prob to the perfered alter
                pref_alter = self.task_dict[node]
                x = 1.7 / n_alters
                y = 1.0/len(self.task_dict)
                maxp_for_best_alter = pow(x,y)                
                dist[pref_alter] = maxp_for_best_alter

                alt_list = list(self.func_alter_devices[node])
                alt_list.remove(pref_alter)

                # generate random prob for the rest of alter
                if n_alters == 2:
                    dist.update(dict(zip(alt_list, [1-maxp_for_best_alter])))
                    # print([1.0-maxp_for_best_alter])
                else:
                    rand_rest = np.random.random(n_alters - 1)
                    # to make the maxp+sum(rand_rest) = 1
                    rand_prob = [e/sum(rand_rest)*maxp_for_best_alter for e in rand_rest]
                    #np.delete(p, np.amax(p))
                    dist.update(dict(zip(alt_list, rand_prob)))
            # save node with its prob
            node_prob_dict[node] = dist

        # these nodes have parents, generate CPT for them
        for node, parent_lst in child_parent.items():
            # parents + this node condProbTable
            condProbTable = self.getCondProbTable(node, parent_lst)
            # save node with its prob
            node_prob_dict[node] = condProbTable
           # print("child node: ", node, " table:", condProbTable)
        
        return node_prob_dict


    def get_BN(self, DAG, child_parent):

        #1. get DAG structure as an arguments        
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
            # print("Parent", node, prob_dist)
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
                    # print("parent child", parent_lst, node, node_prob_dict[node]) 
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
                # print("Netwoerk:", a, b)
        #       print("Network has ", self.network.node_count() , " nodes and ", self.network.edge_count(), " edges")
        return self.network


    def get_permutation_groups(self, parent_node_lst):
        # print("Parents,node", parent_node_lst)
        alter_dev = []
        for n in parent_node_lst:
            alter_dev.append( self.func_alter_devices[n] )
            # list(range(n_att))
        # print("dev for all ", alter_dev)
        alter_perm = itertools.product(*alter_dev)
        # print("alter_perm:", alter_perm)
        permutation = list(dict(zip(parent_node_lst, x)) for x in  alter_perm )
        # print("permutation")   
        # print(permutation)

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
        
        
        condProbTable = []

        n_func_dev = len(self.func_alter_devices[ node ])
        #p^(1/N)
        maxp_for_best_alter = pow(1.7 / n_func_dev,1/len(self.task_dict))
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
        

