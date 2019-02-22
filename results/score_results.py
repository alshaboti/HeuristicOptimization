#!/usr/bin/env python3

from __future__ import division, print_function
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
# In a notebook environment, display the plots inline

# Set some parameters to apply to all plots. These can be overridden
# in each plot if desired
import matplotlib
# Plot size to 14" x 7"
matplotlib.rc('figure', figsize = (14, 7))
# Font size to 14
matplotlib.rc('font', size = 14)
# Do not display top and right frame lines
matplotlib.rc('axes.spines', top = False, right = False)
# Remove grid lines
matplotlib.rc('axes', grid = False)
# Set backgound color to white
matplotlib.rc('axes', facecolor = 'white')


# Define a function for a grouped bar plot
def groupedbarplot(x_data, y_data_list, y_data_names, colors, x_label, y_label, title,yerr):
    _, ax = plt.subplots()
    # Total width for all bars at one x location
    total_width = 0.8
    # Width of each individual bar
    ind_width = total_width / len(y_data_list)
    # This centers each cluster of bars about the x tick mark
    alteration = np.arange(-(total_width/2), total_width/2, ind_width)

    # Draw bars, one category at a time
    for i, y_data in enumerate(y_data_list):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones     
        ax.bar(x_data + alteration[i]
              ,y_data, color = colors[i] \
              ,label = y_data_names[i] \
              ,width = ind_width \
              ,yerr = yerr[i])
       #ax.bar(ind + width, women_means, width, color='y', yerr=women_std)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'upper right')

    plt.setp(ax, xticks = task_len,
            xticklabels = task_len)
    plt.savefig(title + '.png')

def log_scale(value):
    if value > 0:
        return math.log(value)
    return value

if __name__ == '__main__':

    path = './results.csv'
    data = pd.read_csv(path,sep='$')
    task_len = [ 4, 5, 6, 7]

    # get score data
    algorithms_score = ['HC_score','SA_score','GA_score']
    mean_score = data[['task_len'] + algorithms_score ].groupby('task_len').mean()
    mean_score_log = [[log_scale(m) for m in mean_score[alg] ] for alg in algorithms_score]
    std_score = data[['task_len'] + algorithms_score ].groupby('task_len').std()
    # compute standard error 
    stde_score = { alg: [ stdi/math.sqrt(30)*1.96 for stdi in  std_score[alg] ] for alg in algorithms_score}
    stde_score_log = [[log_scale(v) for v in V] for _,V in stde_score.items() ]


    # get time data
    algorithms_time = ['HC_time','SA_time','GA_time']
    mean_time = data[['task_len'] +  algorithms_time].groupby('task_len').mean()
    mean_time_log = [[math.log(m) for m in mean_time[alg] ] for alg in algorithms_time]
    std_time = data[['task_len'] + algorithms_time].groupby('task_len').std()
    # compute standard error 
    stde_time = { alg: [ stdi/math.sqrt(30)*1.96 for stdi in  std_time[alg] ] for alg in algorithms_time}
    stde_time_log = [[log_scale(v) for v in V] for _, V in stde_time.items() ]


    print(mean_time_log)
    print(stde_time_log)
    print(mean_score.index.values)
    # Call the function to create score plot
    groupedbarplot(x_data = mean_score.index.values
                , y_data_list = mean_score_log
                , y_data_names =  ['HC','SA','GA']
                , colors = ['gray', 'blue', 'green']
                , x_label = 'Number of functions in a task'
                , y_label = 'User preference score'
                , title = 'User preference optimization'
                , yerr= stde_score_log
                )

   # Call the function to create plot
    groupedbarplot(x_data = mean_time.index.values
                , y_data_list = mean_time_log
                , y_data_names =  ['HC','SA','GA']
                , colors = ['gray', 'blue', 'green']
                , x_label = 'Number of functions in a task'
                , y_label = 'Time in second'
                , title = 'Elapsed searching time'
                , yerr= stde_time_log
                )

#    plt.show()
