import numpy as np
import random


def get_devices():
    n_devices = 30
    k_capbs = 3
    #return np.random.randint(65, 75, size=(n_devices, k_capbs))
    return np.array([random.sample(range(65,80),k_capbs) for i in range(n_devices)])


def get_task(dev, num_subtask):
    while True:
        task = random.sample(range(65,80),num_subtask)
        num_satisfied_tasks = 0
        for t in task:
            if np.isin(dev,t).any(axis=0).any():#any row and col
                num_satisfied_tasks += 1
        if num_satisfied_tasks == num_subtask:
            return task


def get_init_candidate(devices, task):
    # return list of devices index that have capab to
    # exec task func, first indx for first func etc.
    init_candidate = []
    for f in task:
        for d_number in range(len(devices)):
            if np.isin(devices[d_number],f).any():
                init_candidate.append(d_number)
                break
    return init_candidate


def get_neighbors(cand,devices, task):
    neighbor_list = []
    #for each sub task
    for sub_task_idx in range(len(task)):
        sub_task = task[sub_task_idx]
        # check which devices can exec each subTask.
        dev_idxs = np.where(np.isin(devices,sub_task))[0]
        for alt_dev_idx in dev_idxs:
            if alt_dev_idx != cand[sub_task_idx]:
                new_neighbor = cand.copy()
                new_neighbor[sub_task_idx] = alt_dev_idx
                neighbor_list.append(new_neighbor)
    return  neighbor_list


if __name__ == "__main__":
    devices = get_devices()
    task = get_task(devices,3)
    print(devices)
    print(np.isin(devices,task))
    print(task)
    # print(np.where(np.isin(devices,task)))
    # print(np.isin(devices,task).any(axis=0))
    cand = get_init_candidate(devices,task)
    print(cand)
    print(get_neighbors(cand, devices, task))

