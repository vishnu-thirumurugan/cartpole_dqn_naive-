# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:36:45 2022

@author: oe21s024

plotter

"""

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, epsilons, filename):
    
    # figure and add_subplot function
    fig = plt.figure()
    ax = fig.add_subplot(111, label = "1")
    ax2 = fig.add_subplot(111, label = "2", frame_on = False)
    
    
    # PLOTTING figure1 FOLLOWED BY SETTING X AXIS and setting ticks
    ax.plot(x, epsilons, color = "C0")
    ax.set_xlabel("Training steps", color = "C0")
    ax.set_ylabel("epsilons", color = "C0")
    ax.tick_params(axis = 'x', colors = 'C0')
    ax.tick_params(axis = 'y', colors = 'C0')
    
    
    # plotting running_avg
    N =len(scores)   
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])
    ax2.plot(x, running_avg, color = "C1" )
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("scores", color = "C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis = 'y', colors = 'C1')
    
    plt.savefig(filename)

    
    
    
