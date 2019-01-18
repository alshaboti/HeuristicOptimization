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
#from pomegranate import *
import numpy as np
# Defining the Bayesian Model

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import numpy as np
import pandas as pd
# https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4b_Bayesian_Network_Structure_Learning.ipynb
# https://github.com/pgmpy/pgmpy
# another example: https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb


#1000000 points, dim=10, value: 0:100
# freq  {50: 0, 100:0,150:0,200.0: 1, 250.0: 241,300.0: 11022, 400.0: 400753, 450.0: 367548, 350.0: 117474,
#  500.0: 97764,  550.0: 5130,  600.0: 67, 650:0, 700:0, 800:0,850:0,900:0,950:0}
# rand = random
# rand.seed(4759843)
# np_rand = np.random
# np_rand.seed(94759843)
class JointProbModel:
    """" A Probability model for user preferences"""

    def __init__(self, n):
        self.n_devices = n
        self.devices = []
        self.pair_dist = np.zeros((self.n_devices, self.n_devices), dtype=int)
        self.dims = 10  # device attributes
        # attribute value range
        self.att_bound = [0, 10]
        self.max_edge = math.sqrt(self.dims*pow(self.att_bound[1], 2))
        self.gen_devices()

    def gen_devices(self):
        self.devices = [np.random.randint(self.att_bound[0], self.att_bound[1], self.dims)
                        for i in range(0, self.n_devices)]
        for i in range(0, self.n_devices):
            for j in range(i,self.n_devices):
                self.pair_dist[i][j] = euclidean(self.devices[i], self.devices[j]).astype(int)
                self.pair_dist[j][i] = self.pair_dist[i][j]

    def get_min_dist(self, dev_idx_list):
        list_len = len(dev_idx_list)
        dev_dist = np.ones([list_len,list_len])*float("inf")
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

    def get_score(self, point):
        d = self.get_min_dist(point)
        edges = len(point) - 1
        max_d = edges * self.max_edge
        return 1-(d/max_d), # make it list


class SolutionSpace:
    """"This class generate 2D array of devices (available_devices) where each row represnets a device capability.
    also it generates 2D tasks (task) where each row represents a task subfunctions
    the only condition is that all these tasks should have a devices that are capable to perf them"""
    def __init__(self, n_dev,subtask_list, n_dev_capab, n_sub_task):

        self.subtask_list = subtask_list

        self.n_devices = n_dev
        self.n_capab = n_dev_capab
        self.n_subtask = n_sub_task

        self.subtask_dev = {} # move these two vars here
        self.sol_space_size = 1

        self.gen_devices()
        self.get_subtask_dev()

    def gen_devices(self):
        # return 2D array each row is a device capab/func
        self.available_devices = np.array([random.sample(self.subtask_list, self.n_capab) for i in range(self.n_devices)])
        self.task = self.get_task()

    # def get_init_candidate(self, task):
    #     # return list of devices index that have capab to
    #     # exec task func, first index for first func etc.
    #     init_candidate = []
    #     for f in task:
    #         for d_number in range(len(self.available_devices)):
    #             if np.isin(self.available_devices[d_number], f).any():
    #                 init_candidate.append(d_number)
    #                 break
    #     return init_candidate

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
            task = random.sample(self.subtask_list,self.n_subtask)
            num_satisfied_tasks = 0
            for t in task:
                if np.isin(self.available_devices, t).any(axis=0).any():  # any row and col
                    num_satisfied_tasks += 1
            if num_satisfied_tasks == self.n_subtask:
                return task

    def get_subtask_dev(self):
        # return a dict: key is fun idx, value is a list of dev idx that are cabable to execute the func.

        for f_idx in range(len(self.task)):
            self.subtask_dev[f_idx]=[]

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

    def __init__(self,get_neighbor, get_score):
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
                if(neigh_score > next_score):
                    next_node = neighbor_point
                    next_score = neigh_score
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

    def __init__(self,init_state, problem_model, pref_model):
        Annealer.__init__(self, init_state)
        self.problem_model = problem_model
        self.pref_model = pref_model

    def move(self):
        """"select random neighbor"""
        neighbors = self.problem_model.get_neighbors(self.state)
        # print(neighbors)
        self.state = neighbors[random.randint(0, len(neighbors)-1)]
        # print(self.state)

    def energy(self):
        """" calculate the spanning tree distance """
        e = self.pref_model.get_score(self.state)[0]
        # print(self.state, e)
        return 1-e


class ExhaustiveSearch:
    """" Exhaustive Search class"""

    def __init__(self, subt_dev_dict, get_score):
        self.subt_dev_dict = subt_dev_dict
        self.get_score = get_score

    def run(self):
        max_score = -1
        best_cand = []
        for fun_dev_cand in self.cprod():
            tmp_cand = list(fun_dev_cand.values())
            tmp_score =  self.get_score(tmp_cand)[0]
            if tmp_score > max_score:
                max_score = tmp_score
                best_cand = tmp_cand
        return max_score, best_cand


    def cprod(self):
        """Generate cartesian product"""

        if sys.version_info.major > 2:
            return (dict(zip(self.subt_dev_dict, x)) for x in itertools.product(*self.subt_dev_dict.values()))

        return (dict(itertools.izip(self.subt_dev_dict, x))
                for x in itertools.product(*self.subt_dev_dict.itervalues()))


class GA:
    """ GA algorithm """
    def __init__(self, sol_space, user_pref):
        self.sol_space = sol_space
        self.user_pref = user_pref

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
        self.toolbox.register("evaluate", user_pref.get_score)

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
        f_idx = random.randint(0, len(indv)-1)
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

    def run(self, n=100, max_iteration = 100):
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


class UserPreference:

    def __init__(self, devices, tasks):
        self.devices = devices
        self.tasks = tasks
        #self.pgmpy_test()
        self.pomegranate_test()

    def pgmpy_test(self):

        raw_data = np.array([0] * 30 + [1] * 70)  # Representing heads by 0 and tails by 1
        data = pd.DataFrame(raw_data, columns=['coin'])
        print(data)
        model = BayesianModel()
        model.add_node('coin')

        # Fitting the data to the model using Maximum Likelihood Estimator
        model.fit(data, estimator=MaximumLikelihoodEstimator)
        print(model.get_cpds('coin'))
        print("####################")

    def pomegranate_test(self):
        #mydb = np.array([[1,1,1,1],[1,1,1,1],[0,1,1,1]])
        # [[1,2,8]]*3+[[1,3,4]]*3
        mydb = np.array([[1,1,0,0]]*3+[[1,0,1,1]]*2)#[[1,1,0,1]]*1+[[0,0,1,1]]*1)
        mymodel = BayesianNetwork.from_samples(mydb)

        print(mymodel.node_count())
        print (mydb)

       # mymodel.plot()

# there is no need for none, as you always will provide with value for all devices either used 1 or not 0

        #print( mymodel.probability([[1,1,None,None]]) )
        #print(mymodel.probability([[1, None, 1, None]]))
        #print( mymodel.predict_proba([[1,None,1,None]]) )
        #print( mymodel.predict_proba({}) )

        # print(mymodel.to_json())


def main():
    up = UserPreference([],[])

    # total space = total_devices^n_task_subfunc
    # number of devices
    total_devices = 5
    # number of unique function in each devices
    n_device_capab = 3
    # number of unique functions in each task.
    n_task_subfunc = 3 # 3 # 2
    # char from A to P to represents a sub tasks (functions)
    subtask_pool_list = range(10) #[chr(c) for c in range(65, 75)]

    pref_model = JointProbModel(total_devices)

    sol_space = SolutionSpace(total_devices, subtask_pool_list, n_device_capab, n_task_subfunc)


    print("--------------")
    print("Devices_cababilities: \n ", sol_space.available_devices)
    print("The task:",sol_space.task)
    print("Compitable devices per task index: \n", sol_space.subtask_dev)

    return
    exh_search = ExhaustiveSearch(sol_space.get_subtask_dev()[0], pref_model.get_score).run()

    task = sol_space.task
    print(exh_search[0], " ", exh_search[1])

    # define heuristic algorithms objects
    ga = GA(sol_space, pref_model)
    hc = HillClimbing(sol_space.get_neighbors, pref_model.get_score)

    for n_task_subfunc in [4]:
        print("cap , subt", n_device_capab, n_task_subfunc)
        for i in range(30):

            init_cand = sol_space.get_rand_solution()

            (h_cand, h_score) = hc.climb(init_cand)

            simulated_annealing = TasktoIoTmapingProblem(init_cand, sol_space, pref_model)
            (s_cand, s_score) = simulated_annealing.anneal()

            ga_result = ga.run(n=1000, max_iteration=1000)

            print(1.0 - s_score, " ", h_score, " ", ga_result[1]," ", s_cand, " ", h_cand, " ", ga_result[0][0])


#############################
if __name__ == "__main__":
        main()


################### USER PREFERENCE ##############
# ASSUMING USER HAS 4 DEVICES A,B,C,D
# PREF1: A,B, RATHER THAN A,C.
#
# BUT IF D IS USED THEN USER PREFERE C.
# A,C,D
# DATABASE COUDL BE
# AB,AB,AB,AB,AC,ACD,ACD,ACD,ABD,
# OK OK OK OK NO  OK OK  OK NO
# NO DENOTES NOICE;
# WE CAN ADD ANY EXTRA DEVICES OTHER THAN ABCD BUT THEY SHOULD BE INDEPENDENT
# TO SHOW THAT USER HAS NO PREFERENCE ON THEM LIKE
# A,B,C,D,E,
# AB,AB,ABE,ABE,ACE,ACD,ACD,ACDE,ABDE,
#
#






# print(row)
# print(col)
# print(dist)
# print(dev_dist)
# #print(csr_matrix((dist, (row, col)), shape=(k, k)))
# x=csr_matrix(dev_dist)
# Tcsr = minimum_spanning_tree(x)
# print(Tcsr.toarray())
# total_dist = 0
# for i in range(0, len(Tcsr.toarray())):
#   total_dist += sum(Tcsr.toarray()[i])
# print(total_dist)
    # def gen_model(self):
    #   dist = []
    #   for i in range(0, pow(10,6)):
    #     sel_dev = self.select_dev_rand()
    #     print("\n Slected dev: ", sel_dev)
    #     # print(self.get_min_dist(sel_dev))
    #     d = self.get_min_dist(sel_dev)
    #     dist.append(round(d / 50) * 50)
    #
    #   d = np.array(dist)
    #   # An "interface" to matplotlib.axes.Axes.hist() method
    #   n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
    #                               alpha=0.7, rwidth=0.85)
    #   plt.grid(axis='y', alpha=0.75)
    #   plt.xlabel('Value')
    #   plt.ylabel('Frequency')
    #   plt.title('My Very Own Histogram')
    #   plt.text(23, 45, r'$\mu=15, b=3$')
    #   maxfreq = n.max()
    #   # Set a clean upper y-axis limit.
    #   plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    #   plt.show()
    #   freq = Counter(dist)
    #   return freq
