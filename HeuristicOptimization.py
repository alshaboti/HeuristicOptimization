#!/usr/bin/python3
import numpy as np
import  random
import  itertools

from searchalgorithms import *
from usermodel import User_model

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

    # def get_dev_instance(self, dev_id_lst):
    #     # this one should be similar to usermodel alter_list

    #     alt_list = [str(c) for c in range(self.n_dev_alt)]
    #     return {self.all_available_devices[dev_id]:alt_list[dev_id%self.n_dev_alt] \
    #         for dev_id in dev_id_lst}
        


       
def get_dev_funcs(n_devices, n_functions, \
    min_task_per_devices, min_device_per_function):

    devices = { "d" + str(i) : [] for i in range(n_devices) }
    functions = [ "f" + str(i) for i in range(n_functions) ]
    
    # select number of function randomly
    g,s = max(n_devices,n_functions), min(n_devices,n_functions)
    min_nfun = max(int(g/(2*s)), min_task_per_devices)
    max_nfun = max(int(g*3/s), min_task_per_devices)
    #but no device get more than n_functions
    max_nfun =  min(max_nfun, n_functions)

    n_funct_per_device = [random.randint(min_nfun, \
                 max_nfun) for _ in range(n_devices)]

    while sum(n_funct_per_device) < n_functions:
            n_funct_per_device = [random.randint(min_nfun, \
                 max_nfun) for _ in range(n_devices)]
        
    for i in range(n_devices):
        devices["d"+str(i)].extend( random.sample(functions, \
            n_funct_per_device[i]))
    
    func_alter_devices = { f: [] for f in functions}   
    for f in functions:
        selection_times = 0
        for d, d_fns in devices.items():             
            if f in d_fns:
                func_alter_devices[f].append(d)
                selection_times += 1
        # check if a function doesn't selected and add it to random device
        while selection_times < min_device_per_function:
            dev_idx = random.randint(0,n_devices-1)
            n_funct_per_device[dev_idx] += 1            
            func_alter_devices[f].append("d"+str(dev_idx))
            selection_times += 1

    return devices, functions, func_alter_devices

def main(iteration):
    # number of devices type e.g. door locks, lights, coffee makers
    n_devices = 60
    # all functions in the IoT devices
    n_functions = 30  
    min_task_per_devices = 2 
    min_device_per_function = 2

    # number of funcitons for each device
    devices, functions, func_alter_devices = get_dev_funcs(n_devices,n_functions, \
            min_task_per_devices, min_device_per_function )

    alter_list_count = []
    for f,devs in func_alter_devices.items():
        alter_list_count.append(len(devs))


    n_edges = randint(int(n_functions * 0.25), int(n_functions))
    if n_edges < 1:
        n_edges = 1
    user_model = User_model(nodes = functions, \
                            n_edges = n_edges, \
                            devices = devices, \
                            func_alter_devices = func_alter_devices, \
                            is_gen_task = True)

    print("devices: ",devices)
    print("functions:",functions)
    print("fun:dev: ",func_alter_devices)
    print("TASK: ",user_model.task_dict)
    print("Devices per task list", alter_list_count)

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

    for i in range(30):
        print("internal iteration:", i)
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
        ga_result = ga.run(n=1000, max_iteration=1000)
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

        result = '{0} $ {1} $ {2} $ {3} $ {4} $ {5} $ {6} $ {7} $ {8} $ \
                {9} $ {10} $ {11} $ {12} $ {13} $ {14} $ {15} $ {16} $ {17}'.format( \
                iteration,  \
                i , \
                len(user_model.task_fucs), \
                int( sum(alter_list_count)/len(alter_list_count)), \
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
        print(result)
        with open('data3.txt','a+') as f:
            f.write(result+"\n")             


if __name__ == "__main__":
    start = timer()
    for i in range(10):
        print("external iteration: ", i)
        main(i)
    end = timer()
    print("Over all Elapse time is sec {}".format(end-start))

