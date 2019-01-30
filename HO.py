import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import random
from simanneal import Annealer
import sys, math
import itertools
from timeit import default_timer as timer
# GA
from deap import base
from deap import creator
from deap import tools

from pomegranate import BayesianNetwork
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, State

import networkx as nx
import  random
import  itertools
import numpy as np
# from pomegranate import *
import numpy as np
# Defining the Bayesian Model

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import numpy as np
import pandas as pd

# np.random.seed(5)
# random.seed(5)
randint = random.randint
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
    # this one use a predefined user_prefer_cand and compare the current cand with it.
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


class ProblemDomain:
    """"This class generate an array of devices (available_devices)
    [0 0 1 1 2 2 3 3] where each type repeated by number of alternatives, the value
    represents the function they can do, so capabilities only one for now.
    e.g. device number 2,3 can do task number 1 """

    def __init__(self, n_dev_types,n_dev_alter, subtask_pool_list, n_dev_capab, n_sub_task):

        self.subtask_pool_list = subtask_pool_list
        self.n_dev_types = n_dev_types
        self.n_dev_alt = n_dev_alter
        self.n_total_devices = n_dev_types * n_dev_alter
        self.n_capab = n_dev_capab
        self.n_subtask = n_sub_task

        self.subtask_dev = {}
        self.all_available_devices = self.gen_devices()
        self.task = self.get_task()

        self.prob_domain_size =self.get_subtask_dev()[1]

    def gen_devices(self):
        # return an array with a list of all devices
        # index is the device, value is the function it can do (capabilities).
        # e.g. [0 0 1 1 2 2 3 3], four type of devices each two have same capabilities
        # d0,d1 can do function 0, d4,d5 can do function 2
        return np.array([i//self.n_dev_alt for i in range(self.n_total_devices)])

    def get_all_neighbors(self, cand):
        # Get all other cand that can do the task but differ from current cand by one device.
        neighbor_list = []
        # for each sub task
        for sub_task_idx in range(len(self.task)):
            sub_task = self.task[sub_task_idx]
            # check which devices can exec each subTask.
            dev_idxs = np.where(np.isin(self.all_available_devices, sub_task))[0]
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
                if np.isin(self.all_available_devices, t).any(axis=0).any():  # any row and col
                    num_satisfied_tasks += 1
            if num_satisfied_tasks == self.n_subtask:
                return task


    def get_subtask_dev(self):
        # return a dict: key is fun idx, value is a list of dev idx that are cabable to execute the func.

        for f_idx in range(len(self.task)):
            self.subtask_dev[f_idx] = []

        for f_idx, dev_list in self.subtask_dev.items():
            for d_number in range(len(self.all_available_devices)):
                if np.isin(self.all_available_devices[d_number], self.task[f_idx]).any():
                    self.subtask_dev[f_idx].append(d_number)

        self.prob_domain_size =1
        for f_id, dev_lst in self.subtask_dev.items():
            self.prob_domain_size *= len(dev_lst)

        return self.subtask_dev, self.prob_domain_size

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

            #print("HC: next score", next_score, self.get_score(next_node)[1])

            # if all neighbor score less than the current then end
            if next_score <= current_score:
                return current_node, current_score
            # otherwise jump to the next best neighbor point.
            else:
                current_score = next_score
                current_node = next_node

#https://github.com/perrygeo/simanneal
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
        # if you don't want to see any update in the output
        self.updates=0

    def move(self):
        """"select random neighbor"""
        neighbors = self.problem_model.get_all_neighbors(self.state)
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

    def __init__(self, prob_domain, get_score):
        self.prob_domain = prob_domain
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
                              prob_domain.get_rand_solution)

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
        f_devs_list = self.prob_domain.subtask_dev[f_idx]

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
        # print(self.prob_domain.subtask_dev)
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

# you may not need to use none, as you always will provide
# # with value for all devices either used 1 or not 0

# print( mymodel.probability([[1,1,None,None]]) )
# print(mymodel.probability([[1, None, 1, None]]))
# print( mymodel.predict_proba([[1,None,1,None]]) )
# print( mymodel.predict_proba({}) )

# print(mymodel.to_json())

class RandomDAG:
    def __init__(self, n_nodes, n_edges):

        self.n_nodes = n_nodes
        self.n_edges = n_edges

        if n_edges > n_nodes * (n_nodes - 1):
            self.n_edges = n_nodes * (n_nodes - 1)

        self.randDAG = nx.DiGraph()

    # connected graph req (n-1) edges at least
    # DAG can't be more than n(n-1) edges
    # https://ipython.org/ipython-doc/3/parallel/dag_dependencies.html

    def random_dag(self):

        """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
        # add nodes, labeled 0...nodes:
        for i in range(self.n_nodes):
            self.randDAG.add_node(i)

        child_parent = {}

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
    def __init__(self, n_nodes, n_alters, n_edges, all_avail_dev):

        self.n_nodes = n_nodes
        self.n_alters = n_alters
        self.n_edges = n_edges
        assert (n_edges < (n_nodes-1)*(n_nodes-2)/2), " Can't create DAG with this number of edges!"
        self.all_devices = all_avail_dev
        self.alters_list = self.get_alter_list(n_alters)
        self.network = self.get_BN()
        self.network.bake()

    def get_alter_list(self, n_alter):
        " used in getCondProb and get_score"
        return [str(c) for c in range(n_alter)]

    def getCondProbTable(self, n_var, n_att):
        var = range(n_var)
        var_att_dict = {}
        for i in var:
            var_att_dict[i] = self.alters_list
            #list(range(n_att))

        permutation = list(dict(zip(var_att_dict, x)) for x in itertools.product(*var_att_dict.values()))

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
    def get_BN(self):

        # 1 Build BN DAG structure
        rand_dag = RandomDAG(self.n_nodes, self.n_edges)
        DAG, child_parent =rand_dag.random_dag()


        # for a, bs in DAG.edge.items():
        #     for b in bs.keys():
        #         print(a, "->", b)
      #  print("Key node, value: parents",child_parent)

        # 2 Build BN probability model
        # 2.1 get probabilityDist or conditional prob table
        node_prob_dict = {}
        # these nodes have parents, generate CPT for them
        for node, parent_lst in child_parent.items():
            # parents + this node condProbTable
            condProbTable = self.getCondProbTable(len(parent_lst)+1, self.n_alters)
            # save node with its prob
            node_prob_dict[str(node)] = condProbTable
#            print("Conditional Probability Table: node, parent", node," - ", parent_lst, " \n", condProbTable)

        nodes_list = list(range(self.n_nodes))
        node_with_parent_lst = child_parent.keys()
        node_without_parents = [e for e in nodes_list if e not in node_with_parent_lst]

        # these nodes have no parents so create random prob for them only no conditional here
        for node in node_without_parents:
            p = np.random.random(self.n_alters)
            p /= p.sum()
            dist = {}
            for j in range(self.n_alters):
                dist[str(j)] = p[j]
            # save node with its prob
                node_prob_dict[str(node)] = dist
            #print("Root node: ", node, " dist: ", dist)

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

        # 3 Create BN and add the nodes_state
            self.network = BayesianNetwork("User_pref")
        for node, state in nodes_state.items():
            self.network.add_node(state)
#        print("Network has ", self.network.node_count() , " nodes")

        # 4 Link nodes with edges using nodes_state and DAG.edge
        for a, bs in DAG.edge.items():
            for b in bs.keys():
                self.network.add_edge(nodes_state[a], nodes_state[b])
 #       print("Network has ", self.network.edge_count(), " edges")
        return self.network

    def get_score(self, cand_list):

        can_dev = []

        for node_id in range(self.n_nodes):
            is_added = False
            for node_alter_id in range(self.n_alters):
                if node_id * self.n_alters + node_alter_id in cand_list:
                    can_dev.append(self.alters_list[node_alter_id])
                    is_added = True
            if not is_added:
                can_dev.append(None)


        score = self.network.probability(can_dev)

        return score,can_dev


#############################
def main():
    # user_pref = UserPreference([],[])

    # total space = n_devices^n_task_subfunc
    # number of devices type e.g. door locks, lights, coffee makers
    n_dev_types = 30
    # number of devices for each type, e.g two of each door, light,alarm, coffee maker
    n_dev_alter = 5

    # devices will be device1_alter1 device1_alter2 device2_alter1 device2_alter2 ... etc
    # so when devices 4 selected it means device2_alter2 from BN

    # number of unique function(selected from the subtask_pool_list in each devices
    n_device_capab = 1
    # number of unique functions in each task e.g. may use three devices
    n_task_subfunc = 3 #int(n_dev_types/2)
    # e.g. 0 for doors, 1 for alarm, 2 for light, 3 for coffee maker
    subtask_pool_list = set(range(n_dev_types))




    # using BN as user preference model (static)
    # user_model = Static_User_model()
    # print(user_model.get_score(['d1', 'a1', None, 'c2']))
    #print(network.predict_proba({}))





    for n_task_subfunc in range(3,10):

        prob_domain = ProblemDomain(n_dev_types =n_dev_types, n_dev_alter = n_dev_alter, \
                                    subtask_pool_list=subtask_pool_list, \
                                    n_dev_capab=n_device_capab, n_sub_task= n_task_subfunc)

        # use senthesis BN as user preference model
        # max edges: (n_dev_types-1)*(n_dev_types-2)/4)
        user_model = User_model(n_nodes=n_dev_types, n_alters=n_dev_alter, \
                                n_edges=randint(int(n_dev_types / 4), n_dev_types), \
                                all_avail_dev=prob_domain.all_available_devices)

        ################################

        #print("joint prob \n", user_model.network.probability([None, '1', '1', '0']))

        #exit(0)
        print("--------------")
#        print("Devices cababilities: \n ", prob_domain.all_available_devices)
        print("Devices type {} alternatives of each {} required task {} size {}:".format(n_dev_types, n_dev_alter, prob_domain.task, len(prob_domain.task)))
#        print("Compitable devices per task index: \n", prob_domain.subtask_dev)
        #print("User pref: ",rand_user_pref)

        exh_search = BruteForceSearch(prob_domain.get_subtask_dev()[0], user_model.get_score).run()
        print("Brute Force Search: ", exh_search[0], " ", exh_search[1])




        for i in range(10):

            # define heuristic algorithms objects
            hc = HillClimbing(prob_domain.get_all_neighbors, user_model.get_score)
            ga = GA(prob_domain, user_model.get_score)

            init_cand = prob_domain.get_rand_solution()
            (h_cand, h_score) = hc.climb(init_cand)

            simulated_annealing = TasktoIoTmapingProblem(init_cand, prob_domain, user_model.get_score)
            (s_cand, s_score) = simulated_annealing.anneal()
            #
            ga_result = ga.run(n=1000, max_iteration=1000)
            #
            print("Result ",1.0 - s_score, " ", h_score, " ", ga_result[1][0], " ", s_cand, " ", h_cand, " ", ga_result[0])



if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    print("Elapse time is {} sec ".format(end-start))
