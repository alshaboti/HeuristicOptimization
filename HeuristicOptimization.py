#!/usr/bin/env python3

import numpy as np
import  random
import  itertools

from searchalgorithms import *
from usermodel import User_model
from Results import OutputResult
from timeit import default_timer as timer
# np.random.seed(5)
# random.seed(5)
randint = random.randint
# https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4b_Bayesian_Network_Structure_Learning.ipynb
# https://github.com/pgmpy/pgmpy
# another example: https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb

# from https://github.com/jmschrei/pomegranate/blob/master/examples/bayesnet_asia.ipynb

class ProblemDomain:
    """" """

    def __init__(self, devices, functions, func_alter_devices, task ):
        # devices dict with their capabilities
        self.devices = devices
        # all functions available
        self.functions = functions
        # required task
        self.task = task
        # func dict with their devices 
        self.func_alter_devices = func_alter_devices

    def get_all_neighbors(self, cand):
        # Get all other cand that can do the task but differ from current cand by one device.
        neighbor_list = []
        for idx in range(len(cand)):
            # get the func assigned
            fun = self.task[idx]
            # for each alternative device create a neighbor
            for alt_dev in self.func_alter_devices[fun]:
                alt_cand = cand[:] # clone
                # change to new alter device
                alt_cand[idx] = alt_dev
                neighbor_list.append(alt_cand)

        return neighbor_list

    def get_rand_solution(self):
        # return a random solution 
        # list of dev_idx ordered by task number (i.e. first dev for first task)
        sol = []
        for t in self.task:
            sol.append(random.choice(self.func_alter_devices[t]))

        return sol

 
def main(n_alter_dev_per_func, task_len, n_iteration, output_results):

    user_model = User_model(is_gen_task = True, \
                            n_alter_dev_per_func = n_alter_dev_per_func) 
  
    # functions are nodes, devices are values for BN node
    user_model.build_model(req_task_len= task_len)

    devices = user_model.devices
    functions = user_model.nodes
    func_alter_devices = user_model.func_alter_devices

    print("devices: ",devices)
    print("functions:",functions)
    print("fun:dev: ",func_alter_devices)
    print("TASK: ",user_model.task_dict)
    
    task, up_cand = map(list, zip(*user_model.task_dict.items()) )
    up_score = user_model.get_score( up_cand)[0] 
    #list(user_model.task_dict.values())
    print("best_cand:", up_cand, \
        " score: ", up_score)
   
    prob_domain = ProblemDomain(devices = devices, \
        functions = functions, \
        func_alter_devices = func_alter_devices, \
        task = task)

    start = timer()
    bf = BruteForceSearch(func_alter_devices, task, \
        user_model.get_score).run()
    print("Brute Force Search: ", bf[0], " ", bf[1])
    end = timer()
    bf_time = end-start

    print("Task len: ", task_len, " no alter devices per task", n_alter_dev_per_func)
    # create output file
    for iteration in range(n_iteration):
        print("iteration:", iteration)
        # define heuristic algorithms objects
        start = timer()
        hc = HillClimbing(prob_domain.get_all_neighbors, user_model.get_score)
        init_cand = prob_domain.get_rand_solution()
        (h_cand, h_score) = hc.climb(init_cand)
        end = timer()
        hc_time = end - start
        # print("Elapse time for HC is {} sec.".format(end - start))

        start = timer()
        ga = GA(prob_domain, user_model.get_score)
        ga_result = ga.run(n=100, max_iteration=1000)
        end = timer()
        ga_time = end - start
        # print("Elapse time for GA is {} sec ".format(end - start))

        start = timer()
        simulated_annealing = TasktoIoTmapingProblem(init_cand, \
            prob_domain, user_model.get_score)
        (s_cand, s_score) = simulated_annealing.anneal()
        end = timer()
        sa_time = end - start
        # print("Elapse time for SA is {} sec ".format(end - start))

        result = '{0}${1}${2}${3}${4}${5}${6}${7}${8}${9}${10}${11}${12}${13}${14}${15}${16}'.format( \
                iteration , \
                task_len, \
                n_alter_dev_per_func, \
                up_score, \
                bf[0], \
                1.0 - s_score,  \
                h_score,  \
                ga_result[1][0],  \
                bf_time,
                sa_time, \
                hc_time, \
                ga_time, \
                up_cand, \
                bf[1], \
                s_cand,  \
                h_cand,  \
                ga_result[0]) 
        
        output_results.write_results(result)


if __name__ == "__main__":
    start = timer()
    header = "InterIter$task_len$n_dev_alter$up_score$BrutForce$SA_score$h_score$ga_score$bf_time$sa_time$hc_time$ga_time$up_cand$bf_cand$s_cand$h_cand$ga_cand"
    output_results = OutputResult(file_name="./results/results.csv", \
                                   header_row =header, sep="$")
    n_iteration = 30
    n_alter_dev_per_func = 7     
    experiment_no = 0
    for task_len in range(2,7):
        print("Experiment with task len ", task_len, " and each with ", n_alter_dev_per_func, " alternative devices")

        main( n_alter_dev_per_func,task_len, n_iteration, output_results)        
        
        from_row = experiment_no*n_iteration
        to_row = (experiment_no+1)*n_iteration-1
        # create figures
        output_results.create_figures(from_row, to_row)

    end = timer()
    print("Over all Elapse time is sec {}".format(end-start))

