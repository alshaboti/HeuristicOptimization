#!/usr/bin/env python3

from __future__ import division, print_function
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
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

#task_len$BrutForce$SA_score$h_score$ga_score
path = './results.csv'
data = pd.read_csv(path,sep='$')
#data['dteday'] = pd.to_datetime(daily_data['dteday'])

mean_by_task_len = data[['task_len','BrutForce','SA_score','h_score','ga_score']].groupby('task_len').mean()
std_by_task_len = data[['task_len','BrutForce','SA_score','h_score','ga_score']].groupby('task_len').std()

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
    for i in range(0, len(y_data_list)):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones
        ii = y_data_list[i]
        print("---------------")
        print(ii)
        
        iyerr = yerr[i]
        print(iyerr)
        ax.bar(x_data + alteration[i], y_data_list[i], color = colors[i], \
            label = y_data_names[i], width = ind_width,yerr=yerr[i], edgecolor='r')
       #ax.bar(ind + width, women_means, width, color='y', yerr=women_std)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'upper right')

# Call the function to create plot
groupedbarplot(x_data = mean_by_task_len.index.values
               , y_data_list = [mean_by_task_len['BrutForce'],mean_by_task_len['SA_score'],mean_by_task_len['h_score'],mean_by_task_len['ga_score']]
               , y_data_names =  ['BF','SA','HC','GA']
              , colors = ['#539caf', '#7663b0', '#7622b0', '#7663cc']
               , x_label = 'Number of functions in a task'
               , y_label = 'User preference score'
               , title = 'User preference optimization'
               , yerr= [std_by_task_len['BrutForce'],std_by_task_len['SA_score'],std_by_task_len['h_score'],std_by_task_len['ga_score']])
plt.show()
