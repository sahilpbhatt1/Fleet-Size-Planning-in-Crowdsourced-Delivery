from __future__ import print_function
from docplex.mp.model import Model
import math
import numpy as np
from scipy.spatial import distance
import copy
import random
import Myopic_functions as func

# todo: update below code to get functions from functions file but still get the stats
# Step 3: run algorithm
def solve(stats, data, m, d, base_case, cfa, penalty):
    # Step 3.1:
    driver_attr, demand_attr = {}, {}  # driver attribute a= (sa, ma, oa, time_first_matched, ha, \bar{ha})
    # sa=1 if driver active, ma=1 if driver available, oa: location, ha: active time, \bar{h}_a: fraction
    # demand attribute b = (ob, db, dst, (tbmin, tbmax))
    driver_list = {'act_available': [], 'act_unavailable': [], 'inactive': [], 'exit': []}
    demand_list = {'active': [], 'fulfilled': [], 'expired': []}

    # Collect driver stats
    driver_stats = {'guarantee_met': [], 'guarantee_missed': []}  # stores value of bar_ha above/below threshold W
    demand_stats = {'met': [], 'missed': []}
    driver_log, tot_num_matches = {}, 0

    # driver numbering restarts each iteration. Active available drivers are moved forward but re-numbered
    for t in range(data['T']):
        print("t = %d" % t)
        m[t, 'exit'] = m['exit'][t]
        if t == 0:
            # Initialize driver attrib.
            for i in range(m[0]):
                loc_0 = m[t, i]

                driver_attr[i] = func.Driver(i, 0, 1, loc_0, 0, 0, 0)
                # drivers inactive, available, 0 active time.
                driver_list['inactive'].append(i)
                driver_log[i] = {'all_t': [['t%d' % t, [0, 1, m[t, i], 0, 0, 0]]]}  # , ('t_postdecision', t):

            for i in range(d[0]):  # no. demand from 1 not 0, since 0 means not matched.
                i_loc_o = d[t, i][0][0]  # d[n][t, i][0][0]
                i_loc_d = d[t, i][0][1]
                dst = distance.euclidean(i_loc_o, i_loc_d)
                demand_attr[i+1] = func.Demand(i+1, i_loc_o, i_loc_d, dst, d[t, i][1], d[t, i][1] + data['delta_t'])
                demand_list['active'].append(i+1)
            # I'm keeping track of driver no. to give each driver/order a unique identifier at each iter.

            num_drivers = len(driver_list['inactive']) + len(driver_list['act_available'])
            num_demand = len(demand_list['active'])

        # Step 3.3: solve optimization problem
        xab, z, dual_a, num_matches, x_ab_time, dst, tot_dst = func.solve_opt(data, t, demand_list, driver_list,
                                                            driver_attr, demand_attr, base_case, cfa, penalty)
        tot_num_matches += num_matches
        driver_stats[t, 'dst'] = dst
        driver_stats[t, 'tot_dst'] = tot_dst

        # Step 3.4 - update attr and vf:
        driver_temp, demand_temp = func.update_attr(driver_list, demand_list, xab, driver_attr, demand_attr, t, data,
                                    x_ab_time, stats, driver_log, driver_stats, demand_stats)

        # Step 3.5 - compute pre-decision state
        num_drivers, num_demand = func.predecision_state(num_drivers, num_demand, driver_temp, demand_temp, driver_list,
                                                         demand_list, t, m, d, driver_attr, demand_attr, data, stats,
                                                         driver_log, driver_stats)

        if t == data['T']-1:
            demand_stats['timed_out'] = demand_list['active']
            driver_stats['timed_out'] = []

            for a in driver_list['act_available'] + driver_list['act_unavailable']:
                if (t + 1) * data['t_interval'] - driver_attr[a].time_first_matched >= data['Guaranteed_service'] and \
                        a in driver_list['act_unavailable']:
                    bar_ha = driver_attr[a].bar_ha
                    if bar_ha >= data['W']:
                        driver_stats['guarantee_met'].append([a, bar_ha])
                    else:
                        driver_stats['guarantee_missed'].append([a, bar_ha])
                else:
                    driver_stats['timed_out'].append([a, driver_attr[a].bar_ha])

    return driver_stats, demand_stats, driver_list, demand_list, driver_log, tot_num_matches


def run(inst, base_case, cfa, penalty, T, mu):
    # Step 1: read data
    loadname = 'data/wage_guarantee_test_inst_T=192_%d_mu%d.npz' % (inst, mu)
    data_set = np.load(loadname)
    stats = True

    data = {'T': T, 't_interval': 5, 'delta_t': data_set['delta_t'].tolist(), 'N': 1000,
            'eta': 0.8, 'W': 0.8, 'theta': (1 / (2 * math.sqrt(200))), 'theta1': 1,
            'gamma': 0.9, 'Guaranteed_service': 120,
            'alpha_penalty': 1.0, 'h_int': 5, 'g_int': 5}  # no time window: 400

    m = data_set['m'].tolist()
    d = data_set['d'].tolist()

    # Step 3: Solve same inst for n = 1,..., 500
    driver_stats, demand_stats, driver_list, demand_list, driver_log, tot_num_matches = \
        solve(stats, data, m, d, base_case, cfa, penalty)
    print(driver_stats)
    print(demand_stats)
    print(driver_log[0]['all_t'])
    base_case_str = '_base_case' if base_case else ''
    cfa_str = '_cfa' if cfa else ''
    penalty_str = '_penalty%d' % penalty if penalty > 1 else ''
    name = 'data/sol_T=%d_myopic%s_inst_no%d_mu%d_%s%s_stats_theta1' % (T, penalty_str, inst, mu, base_case_str, cfa_str)

    # print(obj_val_list)
    np.savez(name, demand_stats=demand_stats, driver_stats=driver_stats, driver_log=driver_log,
             driver_list=driver_list, demand_list=demand_list, tot_num_matches=tot_num_matches)
