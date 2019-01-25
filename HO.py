import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import random
from simanneal import Annealer
import sys, math
import itertools
# GA
from deap import base
from deap import creator
from deap import tools

from pomegranate import BayesianNetwork
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, State

import networkx as nx
from random import randint
import  itertools
import numpy as np
# from pomegranate import *
import numpy as np
# Defining the Bayesian Model

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import numpy as np
import pandas as pd


# https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4b_Bayesian_Network_Structure_Learning.ipynb
# https://github.com/pgmpy/pgmpy
# another example: https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb


# 1000000 points, dim=10, value: 0:100
# freq  {50: 0, 100:0,150:0,200.0: 1, 250.0: 241,300.0: 11022, 400.0: 400753, 450.0: 367548, 350.0: 117474,
#  500.0: 97764,  550.0: 5130,  600.0: 67, 650:0, 700:0, 800:0,850:0,900:0,950:0}
# rand = random
# rand.seed(4759843)
# np_rand = np.random
# np_rand.seed(94759843)
class JointProbModel:
    """" A Probability model for user preferences"""

    def __init__(self, n_dev, alter_dev_subtask_list, user_pref_cand):
        self.n_devices = n_dev
        self.devices = []
        self.pair_dist = np.zeros((self.n_devices, self.n_devices), dtype=int)
        self.dims = 10  # device attributes
        # attribute value range
        self.att_bound = [0, 30]
        self.max_edge = math.sqrt(self.dims * pow(self.att_bound[1], 2))
        self.gen_devices()

        self.alter_dev_subtask_list = alter_dev_subtask_list
        self.user_pref_cand = user_pref_cand
        self._max_dist_to_user_pref = self._get_max_dist_to_user_pref()
        #print("MAX dist to UP: ",self._max_dist_to_user_pref)

    def gen_devices(self):
        self.devices = [np.random.randint(self.att_bound[0], self.att_bound[1], self.dims)
                        for i in range(0, self.n_devices)]
        for i in range(0, self.n_devices):
            self.pair_dist[i][i] = 0
            for j in range(i + 1, self.n_devices):
                self.pair_dist[i][j] = euclidean(self.devices[i], self.devices[j]).astype(int)
                self.pair_dist[j][i] = self.pair_dist[i][j]
        #print("pair distance:")
        #print(self.pair_dist)

    def _get_min_dist(self, dev_idx_list):
        list_len = len(dev_idx_list)
        dev_dist = np.ones([list_len, list_len]) * float("inf")
        i = -1
        for di in dev_idx_list:
            i += 1
            j = -1
            for dj in dev_idx_list:
                j += 1
                dev_dist[i][j] = dev_dist[j][i] = self.pair_dist[di][dj]
        Tcsr = minimum_spanning_tree(csr_matrix(dev_dist))

        total_dist = 0
        for i in range(0, len(Tcsr.toarray())):
            total_dist += sum(Tcsr.toarray()[i])
        return total_dist

    # score based on the ecleadian dist btween cand devices attributes.
    def get_score_rand(self, point):
        d = self._get_min_dist(point)
        edges = len(point) - 1
        max_d = edges * self.max_edge
        return 1 - (d / max_d),  # make it list

    # score based on the ecleadian dist btw user pref and current candidate
    def get_score(self, point):
        dist = 0
        for i in range(len(point)):
            dist += self.pair_dist[point[i],self.user_pref_cand[i]]
        #print("dist is :", dist, " max dist is :", self._max_dist_to_user_pref)

        if self._max_dist_to_user_pref == 0:
            # only one solution avaliable then to avoid 0/0
            return 1.0
        score = dist / self._max_dist_to_user_pref

        return (1-score),

    def _get_max_dist_to_user_pref(self):
        max_dist = [ 0 ] * len(self.user_pref_cand)
        for dev_idx in range(len(self.user_pref_cand)):
            #print("dev ", self.user_pref_cand[dev_idx], " for subtask in task index ", dev_idx)
            for alt_dev_idx in range(len(self.alter_dev_subtask_list[dev_idx])):
                d = self.pair_dist[self.user_pref_cand[dev_idx],self.alter_dev_subtask_list[dev_idx][alt_dev_idx]]
             #   print("dist btw ", self.user_pref_cand[dev_idx], self.alter_dev_subtask_list[dev_idx][alt_dev_idx], " is ",d )
              #  print(d)
                if d >  max_dist[dev_idx]:
                    max_dist[dev_idx] = d
            # print("max: ", max_dist[dev_idx])
            # print("------------------")
        return (sum(max_dist))


class SolutionSpace:
    """"This class generate 2D array of devices (available_devices)
    where each row represnets a device capability.
    also it generates 2D tasks (task) where each row represents a task
    subfunctions the only condition is that all these tasks should have
    a devices that are capable to perf them"""

    def __init__(self, n_dev, subtask_pool_list, n_dev_capab, n_sub_task):

        self.subtask_pool_list = subtask_pool_list

        self.n_devices = n_dev
        self.n_capab = n_dev_capab
        self.n_subtask = n_sub_task

        self.subtask_dev = {}
        self.sol_space_size = 1

        self.gen_devices()
        self.get_subtask_dev()

    def gen_devices(self):
        # return 2D array each row is a device capab/func
        self.available_devices = np.array(
            [i//2 for i in range(self.n_devices)])
        self.task = self.get_task()

    def get_neighbors(self, cand):
        neighbor_list = []
        # for each sub task
        for sub_task_idx in range(len(self.task)):
            sub_task = self.task[sub_task_idx]
            # check which devices can exec each subTask.
            dev_idxs = np.where(np.isin(self.available_devices, sub_task))[0]
            for alt_dev_idx in dev_idxs:
                if alt_dev_idx != cand[sub_task_idx]:
                    new_neighbor = cand.copy()
                    new_neighbor[sub_task_idx] = alt_dev_idx
                    neighbor_list.append(new_neighbor)
        return neighbor_list

    def get_task(self):
        # Return a task to work with
        # cond: A task is feasible to be executed by avaliable devices
        while True:
            task = random.sample(self.subtask_pool_list, self.n_subtask)
            num_satisfied_tasks = 0
            for t in task:
                if np.isin(self.available_devices, t).any(axis=0).any():  # any row and col
                    num_satisfied_tasks += 1
            if num_satisfied_tasks == self.n_subtask:
                return task

    def get_subtask_dev(self):
        # return a dict: key is fun idx, value is a list of dev idx that are cabable to execute the func.

        for f_idx in range(len(self.task)):
            self.subtask_dev[f_idx] = []

        for f_idx, dev_list in self.subtask_dev.items():
            for d_number in range(len(self.available_devices)):
                if np.isin(self.available_devices[d_number], self.task[f_idx]).any():
                    self.subtask_dev[f_idx].append(d_number)

        for f_id, dev_lst in self.subtask_dev.items():
            self.sol_space_size *= len(dev_lst)

        return self.subtask_dev, self.sol_space_size

    def get_rand_solution(self):
        # return a random solution
        sol = []
        for f_id, dev_lst in self.subtask_dev.items():
            sol.append(random.choice(dev_lst))

        return sol

    def is_valid_solution(self, point):

        for f in range(len(point)):
            d_idx = point[f]
            if d_idx not in self.subtask_dev[f]:
                return False

        return True


class HillClimbing:
    """"Hill climbing class"""

    def __init__(self, get_neighbor, get_score):
        self.get_neighbor = get_neighbor
        self.get_score = get_score

    def climb(self, init_node):
        current_node = init_node

        current_score = self.get_score(current_node)[0]
        next_score = current_score
        next_node = current_node

        while True:
            neighbors_list = self.get_neighbor(next_node)
            # go through all nieghbors to get the max score (next_score)
            for neighbor_point in neighbors_list:
                neigh_score = self.get_score(neighbor_point)[0]
                # print("neighbor score:", neigh_score)
                if (neigh_score > next_score):
                    next_node = neighbor_point
                    next_score = neigh_score

            print("HC: next score", next_score, self.get_score(next_node)[1])

            # if all neighbor score less than the current then end
            if next_score <= current_score:
                return current_node, current_score
            # otherwise jump to the next best neighbor point.
            else:
                current_score = next_score
                current_node = next_node



class TasktoIoTmapingProblem(Annealer):
    """" Mapping tasks functions to a best combination of devices preferred by user"""

    # rand.seed(58479)
    # Tmax = 10000
    # steps = 10000
    # updates = 1000

    def __init__(self, init_state, problem_model,get_score ):
        Annealer.__init__(self, init_state)
        self.problem_model = problem_model
        self.get_score = get_score

    def move(self):
        """"select random neighbor"""
        neighbors = self.problem_model.get_neighbors(self.state)
        # print(neighbors)
        self.state = neighbors[random.randint(0, len(neighbors) - 1)]
        # print(self.state)

    def energy(self):
        """" calculate the spanning tree distance """
        e = self.get_score(self.state)[0]
        # print(self.state, e)
        return 1 - e


class BruteForceSearch:
    """" Brute Force Search class"""

    def __init__(self, subt_dev_dict, get_score):
        self.subt_dev_dict = subt_dev_dict
        self.get_score = get_score

    def run(self):
        max_score = -1
        best_cand = []
        for fun_dev_cand in self.cprod():
            tmp_cand = list(fun_dev_cand.values())
            tmp_score = self.get_score(tmp_cand)[0]
            if tmp_score > max_score:
                max_score = tmp_score
                best_cand = tmp_cand
                if max_score == 1.0:
                    return max_score, best_cand
        return max_score, best_cand

    def cprod(self):
        """Generate cartesian product"""

        if sys.version_info.major > 2:
            return (dict(zip(self.subt_dev_dict, x)) for x in itertools.product(*self.subt_dev_dict.values()))

        return (dict(itertools.izip(self.subt_dev_dict, x))
                for x in itertools.product(*self.subt_dev_dict.itervalues()))


class GA:
    """ GA algorithm """

    def __init__(self, sol_space, get_score):
        self.sol_space = sol_space
        self.get_score = get_score

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Attribute generator
        #                      define 'attr_bool' to be an attribute ('gene')
        #                      which corresponds to integers sampled uniformly
        #                      from the range [0,1] (i.e. 0 or 1 with equal
        #                      probability)
        # self.toolbox.register("attr_bool", random.randint, 0, 1)

        # Structure initializers
        #                         define 'individual' to be an individual
        #                         consisting of 100 'attr_bool' elements ('genes')
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              sol_space.get_rand_solution)

        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # ----------
        # Operator registration
        # ----------
        # register the goal / fitness function
        self.toolbox.register("evaluate", self.get_score)

        # register the crossover operator
        self.toolbox.register("mate", tools.cxTwoPoint)

        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        self.toolbox.register("mutate", self.flipFunDev, indpb=0.05)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def flipFunDev(self, indv, indpb):
        # select random func to flip its device
        f_idx = random.randint(0, len(indv) - 1)
        # get the available devices for that func
        f_devs_list = self.sol_space.subtask_dev[f_idx]

        if len(f_devs_list) > 1:
            # select dev_id other than the existing one in indv
            d_idx = random.choice(f_devs_list)
            i = 0
            while d_idx == indv[f_idx] or (d_idx not in f_devs_list):
                i += 1
                d_idx = random.choice(f_devs_list)

            indv[f_idx] = d_idx

        return indv

    def run(self, n=100, max_iteration=100):
        # random.seed(64)

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = self.toolbox.population(n=1000)

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.5, 0.2

        # print("Start of evolution")

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0

        # Begin the evolution
        while max(fits) < 10 and g < max_iteration:
            # A new generation
            g = g + 1
            # print("-- Generation %i --" % g)

            # Select the next generation individuals
            # same invd selected more than once! check if okay.
            offspring = self.toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)

                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            # print("  Min %s" % min(fits))
            # print("  Max %s" % max(fits))
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)
        #
        # print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        # print(self.sol_space.subtask_dev)
        return (best_ind, best_ind.fitness.values)



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
        print(cand_list)
        can_dev = []
        for i in range(8): #to do make it general
            if i%2 == 0:
                if i in cand_list:
                    can_dev.append(self.devices[i])
                elif i+1 in cand_list:
                    can_dev.append(self.devices[i+1])
                else:
                    can_dev.append(None)
        print(can_dev)
        return self.network.probability(can_dev),can_dev

# you may not need to use none, as you always will provide
# # with value for all devices either used 1 or not 0

# print( mymodel.probability([[1,1,None,None]]) )
# print(mymodel.probability([[1, None, 1, None]]))
# print( mymodel.predict_proba([[1,None,1,None]]) )
# print( mymodel.predict_proba({}) )

# print(mymodel.to_json())




class RandomDAG:
    def __init__(self):
        self.randDAG = nx.DiGraph()

    # connected graph req (n-1) edges at least
    # DAG can't be more than n(n-1) edges
    # https://ipython.org/ipython-doc/3/parallel/dag_dependencies.html
    def random_dag(self, n_nodes, n_edges):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        child_parent = {}

        if n_edges > n_nodes * (n_nodes - 1):
            self.n_edges = n_nodes * (n_nodes - 1)

        """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
        self.randDAG = nx.DiGraph()
        # add nodes, labeled 0...nodes:
        for i in range(self.n_nodes):
            self.randDAG.add_node(i)

        # to avoid infinit loop, need to have better solution
        round = 1000
        while self.n_edges > 0 and round > 0:
            round -= 1

            a = randint(0, self.n_nodes - 1)
            b = a
            while b == a or self.randDAG.has_edge(a, b):
                b = randint(0, self.n_nodes - 1)
            self.randDAG.add_edge(a, b)
            if nx.is_directed_acyclic_graph(self.randDAG):
                self.n_edges -= 1
                parent = child_parent.get(b)
                if parent is None:
                    parent = [a]
                else:
                    parent.append(a)
                child_parent[b] = parent
                # print(a,"-> ", b)
            else:
                # we closed a loop!
                self.randDAG.remove_edge(a, b)

        return self.randDAG, child_parent


# from https://github.com/jmschrei/pomegranate/blob/master/examples/bayesnet_asia.ipynb
class User_model:

    def getCondProbTable(self, n_var, n_att):
        var = range(n_var)
        var_att_dict = {}
        for i in var:
            var_att_dict[i] = [ str(c) for c in range(n_att)]
            #list(range(n_att))
        if sys.version_info.major > 2:  # python3
            permutation = list(dict(zip(var_att_dict, x)) for x in itertools.product(*var_att_dict.values()))
        else:
            # python2
            permutation = list(dict(itertools.izip(var_att_dict, x))
                               for x in itertools.product(*var_att_dict.itervalues()))

        #print(permutation, ", ", len(permutation), "\n")
        condProbTable = []
        # for each permutation generate prob such that the sum is = 1
        for i in range(int(len(permutation) / n_att)):
            a = np.random.random(n_att)
            a /= a.sum()
            # to make sum of alter prob = 1
            for j in range(n_att):
                condProbRow = list(permutation[i *n_att + j].values())
                condProbRow.append(a[j])
               # print(permutation[i *n_att + j], " = ", a[j])
               # print(condProbRow)
                condProbTable.append(condProbRow)
        return condProbTable

    # n_types: number devices types (e.g. door_lock, camera, etc)
    # n_alter: number of devices from same type
    # n_types: number devices types (e.g. door_lock, camera, etc)
    # n_alter: number of devices from same type
    def get_BN(self, n_nodes, n_alters, n_edges):

        # 1 Build BN DAG structure
        DAG, child_parent = RandomDAG().random_dag(n_nodes, n_edges)

        for a, bs in DAG.edge.items():
            for b in bs.keys():
                print(a, "->", b)
        print("Key node, value: parents",child_parent)

        # 2 Build BN probability model
        # 2.1 get probabilityDist or conditional prob table
        node_prob_dict = {}
        # these nodes have parents, generate CPT for them
        for node, parent_lst in child_parent.items():
            # parents + this node condProbTable
            condProbTable = self.getCondProbTable(len(parent_lst)+1, n_alters)
            # save node with its prob
            node_prob_dict[str(node)] = condProbTable
            print("Conditional Probability Table: node, parent", node," - ", parent_lst, " \n", condProbTable)

        nodes_list = list(range(n_nodes))
        node_with_parent_lst = child_parent.keys()
        node_without_parents = [e for e in nodes_list if e not in node_with_parent_lst]

        # these nodes have no parents so create random prob for them only no conditional here
        for node in node_without_parents:
            p = np.random.random(n_alters)
            p /= p.sum()
            dist = {}
            for j in range(n_alters):
                dist[str(j)] = p[j]
            # save node with its prob
                node_prob_dict[str(node)] = dist
            print("Root node: ", node, " dist: ", dist)

        # 2.2 Create nodes linked to its parent, parent should be processed first.
        # all node state saved to be added to the BN later
        nodes_state = {}
        # all node dist or CPT saved to link child to parents when building child CPT
        nodes_dist = {}

        # start with root nodes (don't have parents then link child to them)
        for node in node_without_parents:
            prob_dist = node_prob_dict[str(node)]
            node_dist = DiscreteDistribution(prob_dist)
            nodes_dist[node] = node_dist
            nodes_state[node] = State(node_dist, name = str(node))
            # remove from nodes_list
            nodes_list.remove(node)


        # rest of the node should have parents
        count = 100
        while len(nodes_list) > 0 and count > 0:
            count -= 1
            for node, parent_lst in child_parent.items():
                # if node's parents already created then it can be created now
                if set(parent_lst).issubset(nodes_state.keys()) and node in nodes_list:

                    node_dist = ConditionalProbabilityTable(node_prob_dict[str(node)] \
                                                      , [nodes_dist[i] for i in parent_lst ])
                    nodes_dist[node] = node_dist
                    nodes_state[node] = State(node_dist, name = str(node))
                    # remove from the node_list
                    nodes_list.remove(node)
        if not nodes_list:
            print("States created for all nodes!")

        # 3 Create BN and add the nodes_state
        network = BayesianNetwork("User_pref")
        for node, state in nodes_state.items():
            network.add_node(state)
        print("Network has ", network.node_count() , " nodes")

        # 4 Link nodes with edges using nodes_state and DAG.edge
        for a, bs in DAG.edge.items():
            for b in bs.keys():
                network.add_edge(nodes_state[a], nodes_state[b])
        print("Network has ", network.edge_count(), " edges")
        return network

def main():
    # user_pref = UserPreference([],[])

    # total space = n_devices^n_task_subfunc
    # number of devices, e.g two of each door, light,alarm, coffee maker
    n_alternatives = 2
    n_nodes = 4
    # devices will be device1_alter1 device1_alter2 device2_alter1 device2_alter2 ... etc
    # so when devices 1,4 selected it means device2_alter2 from BN
    n_devices = n_nodes * n_alternatives
    # number of unique function(selected from the subtask_pool_list in each devices
    n_device_capab = 1
    # number of unique functions in each task e.g. may use three devices
    n_task_subfunc = 3
    # e.g. 0 for doors, 1 for alarm, 2 for light, 3 for coffee maker
    n_subtask_pool = 4

    # using BN as user preference model (static)
    user_model = Static_User_model()
    print(user_model.get_score(['d1', 'a1', None, 'c2']))
    #print(network.predict_proba({}))

    # use senthesis BN as user preference model

    user_model = User_model().get_BN(n_nodes=n_devices/2, n_alters= 2, n_edges=)


    ################################
    subtask_pool_list = range(n_subtask_pool)
    sol_space = SolutionSpace(n_devices, subtask_pool_list, \
                              n_device_capab, n_task_subfunc)

    # select a set of devices as
    rand_user_pref = sol_space.get_rand_solution()
#    pref_model = JointProbModel(n_devices, sol_space.subtask_dev, rand_user_pref)



    print("--------------")
    print("Devices cababilities: \n ", sol_space.available_devices)
    print("The task:", sol_space.task)
    print("Compitable devices per task index: \n", sol_space.subtask_dev)
    print("User pref: ",rand_user_pref)

    # exh_search = BruteForceSearch(sol_space.get_subtask_dev()[0], pref_model.get_score).run()
    # print("Brute Force Search: ", exh_search[0], " ", exh_search[1])


    # define heuristic algorithms objects
    ga = GA(sol_space, user_model.get_score)
    hc = HillClimbing(sol_space.get_neighbors, user_model.get_score)

    for i in range(1):
            init_cand = sol_space.get_rand_solution()
            (h_cand, h_score) = hc.climb(init_cand)
            print(h_cand, h_score)

            simulated_annealing = TasktoIoTmapingProblem(init_cand, sol_space, user_model.get_score)
            (s_cand, s_score) = simulated_annealing.anneal()

            ga_result = ga.run(n=1000, max_iteration=1000)

            print("Result ",1.0 - s_score, " ", h_score, " ", ga_result[1], " ", s_cand, " ", h_cand, " ", ga_result)




#############################
if __name__ == "__main__":

    main()
