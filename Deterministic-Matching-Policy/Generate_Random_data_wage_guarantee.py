# This code generates random instances for dynamic matching
# Go to below link for report
# C:\Users\aliaa\OneDrive - University of Waterloo\Research\Crowdsourced delivery project 2\Latex - Apr

import numpy as np
import itertools
import math
from scipy.spatial import distance

def closest_node(node, loc_set):
    nodes = np.asarray(loc_set)
    closest_ind = distance.cdist([node], nodes).argmin()
    return loc_set[closest_ind]

def sample_path(mu_enter, mu_exit, lambd, grid_size, t_interval, T, loc_set):
    # select sample path 
    d_n = {}
    for t in range(T): 
        # Demand info
        d_t = np.random.poisson(lambd)
        d_n[t] = d_t
        if d_t > 0:
            for i in range(d_t):
                # set random origin and destination locations of drivers on 10x10 grid,
                # random announcement time between 2 epochs
                o_discrete = closest_node([np.random.uniform(0, grid_size[0]), np.random.uniform(0, grid_size[1])],
                                          loc_set)
                d_discrete = closest_node([np.random.uniform(0, grid_size[0]), np.random.uniform(0, grid_size[1])],
                                          loc_set)
                while o_discrete == d_discrete:  # to make sure origin != dest
                    d_discrete = closest_node([np.random.uniform(0, grid_size[0]), np.random.uniform(0, grid_size[1])],
                                              loc_set)
                value_d = np.random.uniform(3, 12)  # 30% of the value of meal, meal value U[10,40]
                d_n[t, i] = [[o_discrete, d_discrete],
                             np.random.uniform(max(0, (t - 1)) * t_interval, t * t_interval), value_d]

    return d_n 
  
if __name__ == '__main__':
    #
    # data = np.load('data/test_inst_T=72.npz')
    # m = data['m'].tolist()
    # d = data['d'].tolist()
    T = 192  # 16 hours --> to cut off the last two at the end
    mu_enter = 2  # changed from 7
    mu_exit = 4
    lambd = 10
    t_interval = 5
    delta_t = t_interval * 18

    theta = 1 / (2 * math.sqrt(200))
    rho_a = [[0, 0.8, 1], [1, 0, 0]]
    rho_b = [[[0, 0.24999999], [0.5, 0.5]], [[0.25, 0.5, 1], [0.25, 0, 0]]]
    min_guarantee = 0.8
    min_wage = 15
    # alpha = 1

    loc_set = [[0.25+(0.5*x), 0.25+(0.5*y)] for x in range(20) for y in range(20)]
    grid_size = (10, 10)
    
    for inst in range(1, 11):
        d = sample_path(mu_enter, mu_exit, lambd, grid_size, t_interval, T, loc_set)

        name = 'data/revised_demand_data_wage_guarantee_test_inst_T=%d_%d_mu%d' % (T, inst, mu_enter)
        np.savez(name, T=T, d=d, mu_enter=mu_enter, mu_exit=mu_exit, lambd=lambd, delta_t=delta_t,
                 t_interval=t_interval, min_wage=min_wage, rho_a=rho_a, rho_b=rho_b)