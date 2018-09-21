import random
import numpy as np
from deap import base
from deap import creator
from deap import tools


# def exhus_search():
#     dic = {"f1":['d1','d2','d3','d4'],"f2": ['d1','d2','d3'],"f3" :['d1','d2']}
#     for k, v in dic.items():
#         for f in k:
#             for


if __name__ == "__main__":
    dic = {"f1":['d1','d2','d3','d4'],"f2": ['d1','d2','d3'],"f3" :['d1','d2']}

    r = random
    r.seed(3432)
    a = np.array([['d1','d2','d3','d4'], ['d1','d2','d3'], ['d1','d2']])

    print(a)
    # exhus_search()
    for f1 in a[0]:
     for f2 in a[1]:
       for f3 in a[2]:
         print(f1,f2,f3)
