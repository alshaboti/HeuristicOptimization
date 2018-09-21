import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import random
from simanneal import Annealer
#1000000 points, dim=10, value: 0:100
# freq  {50: 0, 100:0,150:0,200.0: 1, 250.0: 241,300.0: 11022, 400.0: 400753, 450.0: 367548, 350.0: 117474,
#  500.0: 97764,  550.0: 5130,  600.0: 67, 650:0, 700:0, 800:0,850:0,900:0,950:0}
rand = random
# rand.seed(4759843)
np_rand = np.random
# np_rand.seed(94759843)
import sys
import itertools

class JointProbModel:
    """" A Probability model for user preferences"""

    n_devices = 100 # devices
    devices = []
    pair_dist= np.zeros((n_devices,n_devices),dtype=int)
    dims = 5 # device attributes
    # attribute value range
    att_bound = [0, 10]
    max = dims*att_bound[1]

    def __init__(self, n):
        self.n_devices = n
        self.gen_devices()

    def get_score(self, point):
        d = self.get_min_dist(point)
        return 1-(d/(self.max))

    def gen_devices(self):
        self.devices = [np_rand.randint(self.att_bound[0], self.att_bound[1], self.dims)
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
        #print(Tcsr.toarray())
        total_dist = 0
        for i in range(0, len(Tcsr.toarray())):
            total_dist += sum(Tcsr.toarray()[i])
        return total_dist


class SolutionSpace:

    def __init__(self, n_dev,subtask_list, n_dev_capab, n_sub_task):
        self.subtask_list = subtask_list
        self.n_devices = n_dev
        self.n_capab = n_dev_capab
        self.n_subtask = n_sub_task

        self.gen_devices()

    def gen_devices(self):
        # return np.random.randint(65, 75, size=(n_devices, k_capbs))
        self.available_devices = np.array([rand.sample(self.subtask_list, self.n_capab) for i in range(self.n_devices)])
        self.task = self.get_task()

    def get_init_candidate(self, task):
        # return list of devices index that have capab to
        # exec task func, first indx for first func etc.
        init_candidate = []
        for f in task:
            for d_number in range(len(self.available_devices)):
                if np.isin(self.available_devices[d_number], f).any():
                    init_candidate.append(d_number)
                    break
        return init_candidate

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
        while True:
            task = rand.sample(self.subtask_list,self.n_subtask)
            num_satisfied_tasks = 0
            for t in task:
                if np.isin(self.available_devices, t).any(axis=0).any():  # any row and col
                    num_satisfied_tasks += 1
            if num_satisfied_tasks == self.n_subtask:
                return task


class HillClimbing:
    """"Hill climbing class"""

    def __init__(self,init_node, get_neighbor, get_score):
        self.init_node = init_node
        self.get_neighbor = get_neighbor
        self.get_score = get_score

    def climb(self):
        current_node = self.init_node
        current_score = self.get_score(current_node)
        next_score = current_score
        next_node = current_node

        while True:
            neighbors_list = self.get_neighbor(next_node)
            # go through all nieghbors to get the max score (next_score)
            for neighbor_point in neighbors_list:
                neigh_score = self.get_score(neighbor_point)
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


class ExhaustiveSearch:
    """" Exhaustive Search class"""
    def __init__(self,ful_dict, get_score):
        self.get_neighbor = get_neighbor
        self.get_score = get_score

  def cprod(dictionary):
      """Generate cartesian product"""

      if sys.version_info.major > 2:
          return (dict(zip(dictionary, x)) for x in itertools.product(*dictionary.values()))
    
      return (dict(itertools.izip(dictionary, x))
              for x in itertools.product(*dictionary.itervalues()))

# https://github.com/perrygeo/simanneal


class TasktoIoTmapingProblem(Annealer):
    """" Mapping tasks functions to a best combination of devices preferred by user"""
    rand = random
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
        self.state = neighbors[rand.randint(0, len(neighbors)-1)]
        # print(self.state)

    def energy(self):
        """" calculate the spanning tree distance """
        e = self.pref_model.get_score(self.state)
        # print(self.state, e)
        return 1-e


if __name__ == "__main__":
    # max dist:(max value*dimentntions) 100*10
    total_devices = 100
    n_device_capab = 3
    n_task_subfunc = 3
    for i in range(100):
        subtask_list = range(65, 80) # char from A to P to represents a sub tasks

        pref_model = JointProbModel(total_devices)

        sol_space = SolutionSpace(total_devices,subtask_list, n_device_capab, n_task_subfunc)

        #print(sol_space.available_devices)
        task = sol_space.task
        #print(task)
        init_cand = sol_space.get_init_candidate(task)
        #print(init_cand)
        #print(prob_model.get_min_dist(init_cand), " ", prob_model.get_score(init_cand))

        # for i in sol_space.get_neighbors(init_cand,task):
        #     print(i, prob_model.get_min_dist(i), " ", prob_model.get_score(i))
        simulated_annealing = TasktoIoTmapingProblem(init_cand, sol_space, pref_model)
        simulated_annealing.move()
        s_cand, s_score = simulated_annealing.anneal()

        hc = HillClimbing(init_cand, sol_space.get_neighbors, pref_model.get_score)
        h_cand, h_score = hc.climb()
        print(1.0 - s_score, h_score)




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
