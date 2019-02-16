#!/usr/bin/env python3

from __future__ import division, print_function
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

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

class OutputResult:
    def __init__(self, file_name, header_row, sep="$"):
        self.header_row = header_row
        self.file_name = file_name
        self.path = '/'.join(file_name.split('/')[0:-1])
        self.sep = sep
        
        if not os.path.isfile(self.file_name):
            with open(self.file_name,'w+') as f:
                f.write(self.header_row+"\n") 

        # Define a function to create a boxplot:
    def write_results(self, new_row):
        with open(self.file_name,'a+') as f:
            f.write(new_row+"\n") 
    
    def _boxplot(self, x_data, y_data, base_color, median_color, x_label, y_label, title):
        _, ax = plt.subplots()

        # Draw boxplots, specifying desired style
        ax.boxplot(y_data
                    # patch_artist must be True to control box fill
                    , patch_artist = True
                    # Properties of median line
                    , medianprops = {'color': median_color}
                    # Properties of box
                    , boxprops = {'color': base_color, 'facecolor': base_color}
                    # Properties of whiskers
                    , whiskerprops = {'color': base_color}
                    # Properties of whisker caps
                    , capprops = {'color': base_color})

        # By default, the tick label starts at 1 and increments by 1 for
        # each box drawn. This sets the labels to the ones we want
        ax.set_xticklabels(x_data)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)   
       
    def create_figures(self, from_row, to_row):
        data = pd.read_csv(self.file_name, self.sep)

        score_headers = ["BrutForce" , "SA_score" , "h_score" , "ga_score"]
        x_headers = ["BF", "SA", "HC", "GA"]
        task_len =  data['task_len'][from_row]
        # score figure        
        score_data = [data[x][from_row:to_row].values for x in score_headers]
        x_label = 'Search algorithm (no functions per task = {0}, no alterernative devices ={1} )'.format( \
                    data['task_len'][from_row], \
                    data['n_dev_alter'][from_row])
        y_label = 'User preference probability'
        
        self._boxplot(x_data = score_headers
                , y_data = score_data
                , base_color = '#539caf'
                , median_color = '#297083'
                , x_label = x_label
                , y_label = y_label
                , title = 'User preference score')
        # save figure
        plt.savefig(self.path +'/' +y_label+x_label+str(task_len)+".png")        

        # time figure
        time_headers = ["bf_time" , "sa_time" , "hc_time", "ga_time"  ]
        time_data = [data[x][from_row:to_row].values for x in time_headers]
        x_label = 'Search algorithm (no functions per task = {0}, no alterernative devices ={1} )'.format( \
                    data['task_len'][from_row], \
                    data['n_dev_alter'][from_row])
        y_label = 'Time in seconds'        
        # # Call the function to create plot
        self._boxplot(x_data = x_headers
                , y_data = time_data
                , base_color = '#539caf'
                , median_color = '#297083'
                , x_label = x_label
                , y_label = y_label
                , title = 'Searching time')
        # # save figure
        plt.savefig(self.path +'/' +y_label+x_label+str(task_len)+".png")        
