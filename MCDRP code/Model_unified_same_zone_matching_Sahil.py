from __future__ import print_function
# from sys import getsizeof

# import Compute_optimality_gap as compute_gap
import gurobipy as gp
from gurobipy import GRB
import time
import copy
import random
import pickle
import numpy as np
import math

# last update (Jan 18, 2024)
# - compute MHSP gap for each decision epoch
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def build_model(m_0, N, n_adjacent, n_adjacent_plus, S, D, D_s, scenarios, T, tau_ij, avg_d, gamma, alpha, dist_matrix, beta,
                model_type, enter_exit, disturb, p_dist, forecast_t=5, MHSP_gap=False, penalty_type='TAD'):
    # initiate variables
    # changed delivery deadline to bar_k (make sure to subtract service time)
    Delta, delivery_deadline = 1, 4  # Delta: service time duration (1 epoch) # bar_k 4-1 # delivery deadline
    # min delivery deadline is bar_k - Delta = 1 or 2 (min delivery deadline)
    bar_k, max_tau, x_tot, sol = delivery_deadline - Delta, 2, 0, {}
    sol['time_hip'], sol['time_mcdrp'], sol['gap_hip'], sol['gap_mcdrp'] = [], [], [], []
    Demand = np.sum(np.sum(np.sum(D[tt][i][j] for j in n_adjacent_plus[i]) for i in range(N)) for tt in range(T - 5))
    extended_network = range(N+1) if enter_exit else range(N)

    num_drivers = sum([sum(x) for x in m_0])
    #print('m_0: ', m_0, ' num_drivers: ', num_drivers)
    # for i in range(N):
    #     print('\ni: ', i)
    #     print('Driver types at i: ', [r for r in range(num_drivers) if m_0[i][r]==1]) 

    # create neater lists
    def initial_lists():
        n_adjacent_plus_in, n_adjacent_m_in, n_adjacent_in = {i: [] for i in range(N)}, {i: [] for i in extended_network}, {
            i: [] for i in range(N)}

        if type(avg_d) == list:
            m_0_dummy = round(np.sum(np.sum(avg_d)) * 0.25)
        else:
            m_0_dummy = round(np.sum(np.sum(avg_d[i][j] for j in n_adjacent_plus[i]) for i in range(N)) * 0.25)

        # create neater lists for flow (demand and repositioning)
        if enter_exit:  # and max(extended_network) == N:
            n_adjacent[N] = list(range(N))
            n_adjacent_m = {N: list(n_adjacent[N]) + [N]}
            for i in range(N):
                n_adjacent_m[i] = n_adjacent[i] + [N]
        else:
            n_adjacent_m = copy.copy(n_adjacent)

        for i in range(N):
            for j in n_adjacent_plus[i]:
                n_adjacent_plus_in[j].append(i)
        for i in extended_network:
            for j in n_adjacent_m[i]:
                n_adjacent_m_in[j].append(i)
            if i != N:
                for j in n_adjacent[i]:
                    n_adjacent_in[j].append(i)
        return m_0_dummy, D_s, n_adjacent_m, n_adjacent_m_in, n_adjacent_plus_in, n_adjacent_in

    m_0_dummy, D_s, n_adjacent_m, n_adjacent_m_in, n_adjacent_plus_in, n_adjacent_in = initial_lists()
    sol['D'], sol['D_s'], sol['Demand'], t1 = D, D_s, Demand, time.perf_counter()
    m_ti_transitioning = {tt: [[0 for r in range(num_drivers)] for _ in range(N)] for tt in range(T)}   # new replacement for time.clock()

    # Functions

    def declare_variables(lookahead):  # todo: update return variables 
        # det parameters
        # x_tij (det) # short_lookahead = min(1+Delta, lookahead)  # if in epoch before last, cannot look farther than 1
        keys = [(t2, i, j, r) for t2 in range(1 + Delta) for i in range(N) for j in n_adjacent_plus[i] for r in range(num_drivers)] + \
               [(s, t2, i, j, r) for s in range(scenarios) for t2 in range(1 + Delta, lookahead) for i in range(N)
                for j in n_adjacent_plus[i] for r in range(num_drivers)]
        mdl_xtij = mdl.addVars(keys, vtype=GRB.INTEGER, lb=0, name='x_tij')

        # m_ti (det: 1, stoch: >1)  # add no. of drivers in dummy node
        keys1 = [(t2, i, r) for r in range(num_drivers) for t2 in range(1, 2 + Delta) for i in extended_network]
        keys2 = [(s, t2, i, r) for r in range(num_drivers) for s in range(scenarios) for t2 in range(2 + Delta, lookahead) for i in extended_network]
        keys = keys1 + keys2
        mdl_mti = mdl.addVars(keys, vtype=GRB.INTEGER, lb=0, name='m_ti')

        if model_type in ['UB', 'heatmap']:
            #print('TRUE')
            # u_tij (det: 1: 1+Delta, stoch: >1+Delta) # get to stay or transition
            # transition only to adjacent nodes

            #print('extended_network: ', extended_network); print('n_adjacent_m: ', n_adjacent_m); print(n_adjacent_m == n_adjacent)

            keys1 = [(t2, i, j, r) for r in range(num_drivers) for t2 in range(1, 2 + Delta) for i in extended_network for j in n_adjacent_m[i]]
            keys2 = [(s, t2, i, j, r) for r in range(num_drivers) for s in range(scenarios) for t2 in range(2 + Delta, lookahead)
                     for i in extended_network for j in n_adjacent_m[i]]
            keys = keys1 + keys2
            mdl_utij = mdl.addVars(keys, vtype=GRB.INTEGER, lb=0, name='hat_mtij')  # arrived at time t
        else:
            # u_ti (det: 1: 1+Delta, stoch: >1+Delta) # get to stay or transition
            # transition only to adjacent nodes
            keys1 = [(t2, i, r) for r in range(num_drivers) for t2 in range(1, 2 + Delta) for i in extended_network]
            keys2 = [(s, t2, i, r) for r in range(num_drivers) for s in range(scenarios) for t2 in range(2 + Delta, lookahead)
                     for i in extended_network]
            keys = keys1 + keys2
            mdl_utij = mdl.addVars(keys, vtype=GRB.INTEGER, lb=0, name='hat_mtij')  # arrived at time t

        # d_t,t+kij (det: 1, stoch: >1)
        # todo: if delivery deadline is too short, cannot carry forward demand! create a different code for that
        keys1 = [(t2, t2 + k, i, j) for t2 in range(1, 2 + Delta) for k in range(1, bar_k) for i in range(N)
                 for j in n_adjacent_plus[i]]
        keys2 = [(s, t2, t2 + k, i, j) for s in range(scenarios) for t2 in range(2 + Delta, lookahead)
                 # removed lookahead+1
                 for k in range(1, bar_k) for i in range(N) for j in n_adjacent_plus[i]]
        # check if I need to define d as dependent on travel time
        keys = keys1 + keys2
        mdl_dtij = mdl.addVars(keys, vtype=GRB.CONTINUOUS, lb=0, name='d_ttij')

        # theta_tkij
        keys1 = [(t2, t2 + k, i, j) for t2 in range(1, 2 + Delta) for k in range(1, bar_k) for i in range(N)
                 for j in n_adjacent_plus[i]]
        keys2 = [(s, t2, t2 + k, i, j) for s in range(scenarios) for t2 in range(2 + Delta, lookahead)
                 # removed lookahead+1
                 for k in range(1, bar_k) for i in range(N) for j in n_adjacent_plus[i]]
        keys = keys1 + keys2
        mdl_theta = mdl.addVars(keys, vtype=GRB.CONTINUOUS, lb=0, name='theta_tkij')
        return mdl_xtij, mdl_mti, mdl_utij, mdl_dtij, mdl_theta

    def objective_function(lookahead): 
        # Define the 'match_t' part of the objective function
        #print('m_ti: ', m_ti)
 
        match_t = gp.quicksum(gp.quicksum([mdl_xtij[0, i, j, r] for r in range(num_drivers) if m_ti[i][r] == 1]) * 1.00 +
                              0.8 * gp.quicksum(gp.quicksum([mdl_xtij[bar_t, i, j, r] for r in range(num_drivers)]) for bar_t in range(1, 1 + Delta))
                              for i in range(N) for j in n_adjacent_plus[i])
        
        # Define the 'match_future' part of the objective function 

        match_future = (1 / scenarios) * gp.quicksum(gp.quicksum([mdl_xtij[s, bar_t, i, j, r] for r in range(num_drivers)]) * (0.8 ** bar_t)
                                                     for s in range(scenarios) for bar_t in range(1 + Delta, lookahead)
                                                     for i in range(N) for j in n_adjacent_plus[i])

        # Set and maximize the objective function
        mdl.setObjective(match_t + match_future, GRB.MAXIMIZE)

    def add_constraints(lookahead): 
        def constraint_Aineq1():
            bar_t, travel_t = 0, 1
            # t = 0
            mdl.addConstrs((gp.quicksum(mdl_xtij[bar_t, i, j, r] for j in n_adjacent_plus[i]) <= m_ti[i][r])
                           for i in range(N) for r in range(num_drivers))

            # t = 1  # future t after accounting for travel time is stochastic since d is
            mdl.addConstrs((gp.quicksum(mdl_xtij[bar_t, i, j, r] for j in n_adjacent_plus[i]) <= mdl_mti[bar_t, i, r])
                           for bar_t in range(1, 1 + Delta) for i in range(N) 
                           for r in range(num_drivers))

            # future t (stoch)
            mdl.addConstrs((gp.quicksum(mdl_xtij[s, bar_t, i, j, r] for j in n_adjacent_plus[i]) <= mdl_mti[bar_t, i, r])
                           for s in range(scenarios)
                           for i in range(N) for bar_t in range(1 + Delta, 2 + Delta) 
                           for r in range(num_drivers))

            mdl.addConstrs((gp.quicksum(mdl_xtij[s, bar_t, i, j, r] for j in n_adjacent_plus[i]) <= mdl_mti[s, bar_t, i, r])
                           for s in range(scenarios) for i in range(N)
                           for bar_t in range(2 + Delta, lookahead) 
                           for r in range(num_drivers))

        def constraint_Aineq2():
            # total matching <= demand
            # at t=0 (det) bar_k global variable for delivery deadline
            # todo: maybe change to sum instead of gp.quicksum, since they're parameters
            mdl.addConstrs((gp.quicksum([mdl_xtij[0, i, j, r] for r in m_ti[i]]) <= gp.quicksum(d_t[bar_t][i][j] for bar_t in
                                                range(tt + tau_ij[i][j], tt + bar_k)) + D[tt][i][j])
                           for i in range(N) for j in n_adjacent_plus[i])

            # t=1, d_ttij (det)
            mdl.addConstrs((gp.quicksum([mdl_xtij[bar_t, i, j, r] for r in range(num_drivers)]) <= gp.quicksum(
                mdl_dtij[bar_t, bar_t + k, i, j] for k in range(tau_ij[i][j], bar_k))
                            + D[tt + bar_t][i][j]) for i in range(N)
                           for j in n_adjacent_plus[i] for bar_t in range(1, 1 + Delta))

            # future t (det carried forward demand)
            mdl.addConstrs((gp.quicksum([mdl_xtij[s, bar_t, i, j, r] for r in range(num_drivers)]) <= gp.quicksum(mdl_dtij[bar_t, bar_t + k, i, j]
                                                                    for k in range(tau_ij[i][j], bar_k)) +
                            D_s[s][tt + bar_t][i][j]) for s in range(scenarios)
                           for i in range(N) for j in n_adjacent_plus[i] for bar_t in range(1 + Delta, 2 + Delta))

            # future t (stoch)
            mdl.addConstrs((gp.quicksum([mdl_xtij[s, bar_t, i, j, r] for r in range(num_drivers)]) <= gp.quicksum(mdl_dtij[s, bar_t, bar_t + k, i, j]
                                                                    for k in range(tau_ij[i][j], bar_k)) +
                            D_s[s][tt + bar_t][i][j]) for s in range(scenarios)
                           for i in range(N) for j in n_adjacent_plus[i] for bar_t in range(2 + Delta, lookahead))

        def constraint_Aeq2():
            # t = 0, 0+Delta (det)
            mdl.addConstrs((mdl_dtij[bar_t, bar_t + bar_k - 1, i, j] == D[tt + (bar_t - 1)][i][j] -
                            mdl_theta[bar_t, bar_t + bar_k - 1, i, j])
                           for i in range(N) for j in n_adjacent_plus[i] for bar_t in range(1, 2 + Delta))

            # stoch dt, t+(bar_k -1)
            mdl.addConstrs((mdl_dtij[s, bar_t, bar_t + bar_k - 1, i, j] == D_s[s][tt + (bar_t - 1)][i][j] -
                            mdl_theta[s, bar_t, bar_t + bar_k - 1, i, j])
                           for i in range(N) for j in n_adjacent_plus[i]
                           for bar_t in range(2 + Delta, lookahead) for s in range(scenarios))

            # dt, t+k
            mdl.addConstrs((mdl_dtij[1, 1 + k, i, j] == d_t[tt + k + 1][i][j] - mdl_theta[1, 1 + k, i, j])
                           for i in range(N) for j in n_adjacent_plus[i]
                           for k in range(tau_ij[i][j], bar_k - 1))

            mdl.addConstrs((mdl_dtij[bar_t, bar_t + k, i, j] == mdl_dtij[bar_t - 1, bar_t + k, i, j] -
                            mdl_theta[bar_t, bar_t + k, i, j])
                           for i in range(N) for j in n_adjacent_plus[i] for k in range(tau_ij[i][j], bar_k - 1)
                           for bar_t in range(1 + Delta, 2 + Delta))

            mdl.addConstrs((mdl_dtij[s, bar_t, bar_t + k, i, j] == mdl_dtij[bar_t - 1, bar_t + k, i, j] -
                            mdl_theta[s, bar_t, bar_t + k, i, j])
                           for s in range(scenarios) for i in range(N) for j in n_adjacent_plus[i]
                           for k in range(tau_ij[i][j], bar_k - 1) for bar_t in range(2 + Delta, 3 + Delta))

            mdl.addConstrs((mdl_dtij[s, bar_t, bar_t + k, i, j] == mdl_dtij[s, bar_t - 1, bar_t + k, i, j]
                            - mdl_theta[s, bar_t, bar_t + k, i, j])
                           for s in range(scenarios) for i in range(N) for j in n_adjacent_plus[i]
                           for k in range(tau_ij[i][j], bar_k - 1) for bar_t in range(3 + Delta, lookahead))

        def constraint_Aineq3():
            # t=1: d is a parameter
            mdl.addConstrs((mdl_theta[bar_t, bar_t + k, i, j] >= gp.quicksum([mdl_xtij[bar_t-1, i, j, r] for r in range(num_drivers)]) -
                            gp.quicksum(d_t[kk][i][j] for kk in range(tau_ij[i][j] + tt, tt + k + 1)))
                           for i in range(N) for j in n_adjacent_plus[i]
                           for k in range(tau_ij[i][j], bar_k) for bar_t in [1])

            # t>=2: d is a variable
            mdl.addConstrs((mdl_theta[bar_t, bar_t + k, i, j] >=
                            gp.quicksum([mdl_xtij[bar_t-1, i, j, r] for r in range(num_drivers)]) - gp.quicksum(mdl_dtij[bar_t - 1, bar_t - 1 + kk, i, j]
                                                                    for kk in range(tau_ij[i][j], k + 1)))
                           for i in range(N) for j in n_adjacent_plus[i]
                           for k in range(tau_ij[i][j], bar_k) for bar_t in range(2, 2 + Delta))

            # if tt < T - 2:  # if we're not at last epoch (stoch)
            # dt-1 is det not stoch
            mdl.addConstrs((mdl_theta[s, bar_t, bar_t + k, i, j] >= gp.quicksum([mdl_xtij[s, bar_t-1, i, j, r] for r in range(num_drivers)]) -
                            gp.quicksum(mdl_dtij[bar_t - 1, bar_t - 1 + kk, i, j]
                                        for kk in range(tau_ij[i][j], k + 1)))
                           for i in range(N) for j in n_adjacent_plus[i] for k in range(tau_ij[i][j], bar_k)
                           for bar_t in range(2 + Delta, 3 + Delta) for s in range(scenarios))

            mdl.addConstrs((mdl_theta[s, bar_t, bar_t + k, i, j] >= gp.quicksum([mdl_xtij[s, bar_t-1, i, j, r] for r in range(num_drivers)]) -
                            gp.quicksum(mdl_dtij[s, bar_t - 1, bar_t - 1 + kk, i, j]
                                        for kk in range(tau_ij[i][j], k + 1)))
                           for i in range(N) for j in n_adjacent_plus[i] for k in range(tau_ij[i][j], bar_k)
                           for bar_t in range(3 + Delta, lookahead) for s in range(scenarios))

        def constraint_Aeq3():
            # Aeq 1, flow balance, no. repositioning drivers = no. unmatched drivers
            # t = 0 (det)   # no. unmatched + no. matched = no. avail. drivers
            bar_t, travel_t = 0, 1
            mdl.addConstrs(gp.quicksum(mdl_utij[bar_t + travel_t, i, j1, r] for j1 in n_adjacent_m[i])
                            + (gp.quicksum(mdl_xtij[bar_t, i, j, r] for j in n_adjacent_plus[i]) if i != N else 0)
                            == m_ti[i][r] for i in extended_network for r in range(num_drivers))
 
            # t=1
            mdl.addConstrs((gp.quicksum(mdl_utij[bar_t + travel_t, i, j1, r] for j1 in n_adjacent_m[i]) +
                            (gp.quicksum(mdl_xtij[bar_t, i, j, r] for j in n_adjacent_plus[i]) if i != N else 0)
                            == mdl_mti[bar_t, i, r]) for i in extended_network
                           for bar_t in range(1, 1 + Delta) for r in range(num_drivers))

            # future t (stoch) mti det
            mdl.addConstrs((gp.quicksum(mdl_utij[s, bar_t + travel_t, i, j1, r] for j1 in n_adjacent_m[i]) +
                            (gp.quicksum(mdl_xtij[s, bar_t, i, j, r] for j in n_adjacent_plus[i]) if i != N else 0)
                            == mdl_mti[bar_t, i, r]) for r in range(num_drivers)
                           for s in range(scenarios) for i in extended_network for bar_t in
                           range(1 + Delta, 2 + Delta))

            # future t (stoch)
            mdl.addConstrs((gp.quicksum(mdl_utij[s, bar_t + travel_t, i, j1, r] for j1 in n_adjacent_m[i]) +
                            (gp.quicksum(mdl_xtij[s, bar_t, i, j, r] for j in n_adjacent_plus[i]) if i != N else 0)
                            == mdl_mti[s, bar_t, i, r]) for r in range(num_drivers)
                           for s in range(scenarios) for i in extended_network for bar_t in
                           range(2 + Delta, lookahead - 1))

        def constraint_Aeq4():
            # t=0 (det)d
            mdl.addConstrs((mdl_mti[bar_t, i, r] == (gp.quicksum(mdl_xtij[bar_t - tau_ij[j1][i], j1, i, r] for j1 in
                                                              n_adjacent_plus_in[i] if
                                                              bar_t - tau_ij[j1][i] >= 0) if i != N else 0)
                            + gp.quicksum(mdl_utij[bar_t, j, i, r] for j in n_adjacent_m_in[i])) for r in range(num_drivers)
                           for i in extended_network for bar_t in range(1, 2 + Delta))

            # if tt < T-2:
            mdl.addConstrs((mdl_mti[s, bar_t, i, r] == (gp.quicksum(mdl_xtij[bar_t - tau_ij[j1][i], j1, i, r] 
                            for j1 in n_adjacent_plus_in[i] if 0 <= bar_t - tau_ij[j1][i] < 1 + Delta)
                                                     if i != N else 0)  # sum over right variables
                            + (gp.quicksum(mdl_xtij[s, bar_t - tau_ij[j1][i], j1, i, r] for j1 in n_adjacent_plus_in[i]
                                           if bar_t - tau_ij[j1][i] >= 1 + Delta) if i != N else 0) +
                            gp.quicksum(mdl_utij[s, bar_t, j, i, r] for j in n_adjacent_m_in[i])) for r in range(num_drivers) for s in range(scenarios)
                           for i in extended_network for bar_t in range(2 + Delta, 3 + Delta))

            # t > 1 (stoch)  # above constraint can include this constraint if bar_t up to lookahead
            mdl.addConstrs((mdl_mti[s, bar_t, i, r] == (gp.quicksum(mdl_xtij[s, bar_t - tau_ij[j1][i], j1, i, r] 
                            for j1 in n_adjacent_plus_in[i] if bar_t - tau_ij[j1][i] >= 0) if i != N else 0)
                            + gp.quicksum(mdl_utij[s, bar_t, j, i, r] for j in n_adjacent_m_in[i]))
                           for r in range(num_drivers) for s in range(scenarios) for i in
                           extended_network for bar_t in range(3 + Delta, lookahead))

        constraint_Aineq1()  # Aineq 1, total matching <= drivers available
        constraint_Aineq2()  # total matching <= demand
        constraint_Aeq2()  # carried forward demand
        constraint_Aineq3()  # linearize max term in carried forward demand
        if model_type in ['UB', 'heatmap']:
            constraint_Aeq3()  # flow balance, no. repositioning drivers = no. unmatched drivers
            constraint_Aeq4()  # flow balance, available drivers at t+1 = no. with dest at i + no. that reposition to i

    def update_variables(d_t, m_ti, D):
        sol[tt, 'z'] = mdl.ObjVal
        sol['time_mcdrp'].append(mdl.Runtime)
        sol['gap_mcdrp'].append(mdl.MIPGap)
        sol[tt, 'xtij'], sol[tt, 'dtij'], t_matches = {}, {}, 0

        for i in range(N): 
            for j in n_adjacent_plus[i]:
                #print('i: ', i, ' j: ', j, ' num_drivers: ', num_drivers)
                #print('gp.quicksum([mdl_xtij[0, i, j, r].X for r in range(num_drivers)]): ', gp.quicksum([mdl_xtij[0, i, j, r].X for r in range(num_drivers)]))

                num_matches = sum([mdl_xtij[0, i, j, r].X for r in range(num_drivers)])
                sol[tt, 'xtij'][i, j] = num_matches
                t_matches += num_matches

            #print('i: ', i); print('matches with i at time 0: ', sum([sol[tt, 'xtij'][i, j] for j in n_adjacent_plus[i]])); print()

        #print('mdl_xtij[0, i, j, 0].X: ', sum([mdl_xtij[0, i, j, 0].X for i in range(N) for j in n_adjacent_plus[i]]))

        #print('mdl_xtij[0, i, j, 1].X: ', sum([mdl_xtij[0, i, j, 1].X for i in range(N) for j in n_adjacent_plus[i]]))

        print('no. of matches at t=%d is %d' % (tt, t_matches))

        num_repos_0 = sum([mdl_utij[0+1, i, j, r].X for i in range(N) for j in n_adjacent[i] for r in range(num_drivers)])
        print('num_repos_0: ', num_repos_0)

        print('lookahead: ', lookahead, 'scenarios: ', scenarios)

        # for t in range(1, 1+Delta): 
        #     num_matches = sum([mdl_xtij[t, i, j, r].X for i in range(N) for j in n_adjacent_plus[i] for r in range(num_drivers)])
        #     num_repos = sum([mdl_utij[t+1, i, j, r].X for i in range(N) for j in n_adjacent[i] for r in range(num_drivers)])
        #     print('t: ', t, ' no of drivers at beginning: ', sum([mdl_mti[t, i, r].X for i in range(N) for r in range(num_drivers)]), ' no matches: ', num_matches, ' num_repos: ', num_repos)
            
        # for r in range(num_drivers):  
        #     print('R EQUALS: ', r)
        #     for t in range(1, 1+Delta):    
        #         print(sum(mdl_utij[t + 1, i, j1, r].X for i in range(N) for j1 in n_adjacent[i])) 
                
        #         print(sum(mdl_xtij[t, i, j, r].X for i in range(N) for j in n_adjacent_plus[i]) if i != N else 0) 

        #         print(sum(mdl_mti[t, i, r].X for i in range(N))) 

        #         num_matches = sum([mdl_xtij[t, i, j, r].X for i in range(N) for j in n_adjacent_plus[i]])
        #         num_repos = sum([mdl_utij[t+1, i, j, r].X for i in range(N) for j in n_adjacent[i]])
        #         print('t IS: ', t, ' no of drivers at beginning: ', sum([mdl_mti[t, i, r].X for i in range(N)]), ' no matches: ', num_matches, ' num_repos: ', num_repos)
              
        # for t in range(1+Delta, lookahead): 
        #     num_matches = 1/scenarios * sum([mdl_xtij[s, t, i, j, r].X for i in range(N) for j in n_adjacent_plus[i] for s in range(scenarios) for r in range(num_drivers)])
        #     print('t: ', t, ' av no matches: ', num_matches)

        print('SOLUTION\n')

        for t in range(1): 
            num_matches = sum([mdl_xtij[t, i, j, r].X for r in range(num_drivers) for i in range(N) for j in n_adjacent_plus[i]])
            num_repos = sum([mdl_utij[t+1, i, j, r].X for r in range(num_drivers) for i in range(N) for j in n_adjacent[i]])
            print('t: ', t, ' no of drivers at beginning: ', sum([sum(m_ti[i]) for i in range(N)]), ' no matches: ', num_matches, ' num_repos: ', num_repos)
            #print('matches: ', [mdl_xtij[t, i, j, r] for r in range(num_drivers) for i in range(N) for j in n_adjacent_plus[i] if mdl_xtij[t, i, j, r].X == 1])

        for t in range(1, 1+Delta): 
            num_matches = sum([mdl_xtij[t, i, j, r].X for r in range(num_drivers) for i in range(N) for j in n_adjacent_plus[i]])
            num_repos = sum([mdl_utij[t+1, i, j, r].X for r in range(num_drivers) for i in range(N) for j in n_adjacent[i]])
            print('t: ', t, ' no of drivers at beginning: ', sum([mdl_mti[t, i, r].X for r in range(num_drivers) for i in range(N)]), ' no matches: ', num_matches, ' num_repos: ', num_repos)
            #print('matches: ', [mdl_xtij[t, i, j, r] for r in range(num_drivers) for i in range(N) for j in n_adjacent_plus[i] if mdl_xtij[t, i, j, r].X == 1])
 
        # for bar_t in range(1, 1+Delta): 
        #     for r in range(num_drivers): 
        #         print('r: ', r, ' mdl_mti[bar_t, i, r]: ', sum(mdl_mti[bar_t, i, r].X for i in range(N)))
        #         print('x: ', sum(mdl_xtij[bar_t - tau_ij[j1][i], j1, i, r].X for i in range(N) for j1 in
        #                                                     n_adjacent_plus_in[i] if
        #                                                     bar_t - tau_ij[j1][i] >= 0) if i != N else 0) 
                
        #         print('u: ', sum(mdl_utij[bar_t, j, i, r].X for i in range(N) for j in n_adjacent_m_in[i]))

        #     # mdl.addConstrs((mdl_mti[bar_t, i, r] == (gp.quicksum(mdl_xtij[bar_t - tau_ij[j1][i], j1, i, r] for j1 in
        #     #                                                   n_adjacent_plus_in[i] if
        #     #                                                   bar_t - tau_ij[j1][i] >= 0) if i != N else 0)
        #     #                 + gp.quicksum(mdl_utij[bar_t, j, i, r] for j in n_adjacent_m_in[i])) for r in range(num_drivers)
        #     #                for i in extended_network for bar_t in range(1, 2 + Delta))


        for t in range(1+Delta, 2+Delta): 
            s = 0 
            num_matches = sum([mdl_xtij[s, t, i, j, r].X for r in range(num_drivers) for i in range(N) for j in n_adjacent_plus[i]])
            num_repos = sum([mdl_utij[s, t+1, i, j, r].X for r in range(num_drivers) for i in range(N) for j in n_adjacent[i]])
            print('t: ', t, ' no of drivers at beginning: ', sum([mdl_mti[t, i, r].X for r in range(num_drivers) for i in range(N)]), ' no matches: ', num_matches, ' num_repos: ', num_repos)
            #print('matches: ', [mdl_xtij[s, t, i, j, r] for r in range(num_drivers) for i in range(N) for j in n_adjacent_plus[i] if mdl_xtij[s, t, i, j, r].X == 1])

        for t in range(2+Delta, lookahead-1): 
            s = 0 
            num_matches = sum([mdl_xtij[s, t, i, j, r].X for r in range(num_drivers) for i in range(N) for j in n_adjacent_plus[i]])
            num_repos = sum([mdl_utij[s, t+1, i, j, r].X for r in range(num_drivers) for i in range(N) for j in n_adjacent[i]])
            print('t: ', t, ' no of drivers at beginning: ', sum([mdl_mti[s, t, i, r].X for r in range(num_drivers) for i in range(N)]), ' no matches: ', num_matches, ' num_repos: ', num_repos)
            
        for t in range(lookahead-1, lookahead): 
            s = 0 
            num_matches = sum([mdl_xtij[s, t, i, j, r].X for r in range(num_drivers) for i in range(N) for j in n_adjacent_plus[i]]) 
            print('t: ', t, ' no of drivers at beginning: ', sum([mdl_mti[s, t, i, r].X for r in range(num_drivers) for i in range(N)]), ' no matches: ', num_matches)
        


        # for r in range(1): 
        #     print('\ndriver: ', r)

        #     for t in range(1+Delta): 
        #         print('\nt: ', t)
        #         print('no matches: ', sum([mdl_xtij[t, i, j, r].X for i in range(N) for j in n_adjacent_plus[i]]))
        #         print('no repos: ', sum([mdl_utij[t+1, i, j, r].X for i in range(N) for j in n_adjacent[i]]))
            
        #     for s in range(scenarios): 
        #         print('\nscenario: ', s)    
        #         for t in range(1+Delta, lookahead): 
        #             print('\nt: ', t); print('no matches: ', sum([mdl_xtij[s, t, i, j, r].X for i in range(N) for j in n_adjacent_plus[i]]))
        #             if t+1 < lookahead: 
        #                 print('no repos: ', sum([mdl_utij[s, t+1, i, j, r].X for i in range(N) for j in n_adjacent[i]]))

        #     no_matches = sum([sum([mdl_xtij[t, i, j, r].X for i in range(N) for j in n_adjacent_plus[i]]) for t in range(1+Delta)])
            
        #     #print(sum([sum([mdl_utij[t+1, i, j, r].X for i in range(N) for j in n_adjacent[i]]) for t in range(1+Delta)]))

        #     av_no_matches_s = 1/scenarios * sum([sum([mdl_xtij[s, t, i, j, r].X for i in range(N) for j in n_adjacent_plus[i]]) for t in range(1+Delta, lookahead) for s in range(scenarios)]) 

        #     print('av_no_matches: ', av_no_matches_s)

        #     tot_matches = no_matches + av_no_matches_s

        #     utilization = tot_matches/(lookahead)

        #     print('utilization: ', utilization)

        sol[tt, 'd_t'], sol[tt, 'm_t'], sol[tt, 'D'] = copy.deepcopy(d_t), copy.deepcopy(m_ti), D[tt]

        x_t = {i: {j: {r: int(round(mdl_xtij[0, i, j, r].X)) for r in range(num_drivers)} for j in n_adjacent_plus[i]} for i in range(N)} 

        if model_type in ['UB', 'heatmap']:
            u_tij = {i: {j: {r: int(round(mdl_utij[1, i, j, r].X)) for r in range(num_drivers)} for j in n_adjacent_m[i]} for i in extended_network} 
        else:  # keep name u_tij but it's actually not indexed by j
            u_tij = None #{i: int(round(mdl_utij[1, i].X)) for i in extended_network}

        sol[tt, 'x_t'], sol[tt, 'u_tij'] = x_t, u_tij
        d_tt = {tt + 1 + t_bar: {i: {j: 0 for j in n_adjacent_plus[i]} for i in range(N)}
                for t_bar in range(1, bar_k)}

        for i in range(N):
            for j in n_adjacent_plus_in[i]:
                for r in range(num_drivers): 
                    m_ti_transitioning[tt + tau_ij[j][i]][i][r] += x_t[j][i][r]

        sol[tt, 'm_ti_matching'] = m_ti_transitioning[tt + 1] 

        for i in range(N):
            for j in n_adjacent_plus[i]:
                fulfilled_demand = sum([x_t[i][j][r] for r in range(num_drivers)]) 

                for t_bar in range(tt + tau_ij[i][j], tt + bar_k):
                    if t_bar >= tt + tau_ij[i][j] + 1:
                        d_tt[t_bar][i][j] += max(d_t[t_bar][i][j] - fulfilled_demand, 0)
                    if d_t[t_bar][i][j] <= fulfilled_demand:
                        fulfilled_demand -= d_t[t_bar][i][j]
                    else:
                        fulfilled_demand = 0
                d_tt[tt + bar_k][i][j] += D[tt][i][j] - fulfilled_demand

        d_t = d_tt
        return x_t, u_tij, d_t, t_matches, m_ti_transitioning

    optimality_gap_list = []
    for tt in range(T - 5): 
        print(color.BOLD + '\nCURRENT TIME PERIOD: ', tt, color.END)
          
        for i in range(N):
            print('\ni: ', i)
            if tt > 0:
                print('Driver types at i: ', [r for r in range(num_drivers) if m_ti[i][r]==1]) 
            else:
                print('Driver types at i: ', [r for r in range(num_drivers) if m_0[i][r]==1]) 

        print('t = %d' % tt)
        bar_t = T - tt  # changed from  10 to 5
        lookahead = min(bar_t, forecast_t)  # todo: check if this is needed
        # initiate carried forward demand
        if tt == 0:
            m_ti, d_t = m_0 + [m_0_dummy], {}  # transitioning matches from previous epochs
            for t2 in range(1 + tt, bar_k + tt):  # carried forward demand at first epoch is 0
                d_t[t2] = {}  # range of t
                for i in range(N):
                    d_t[t2][i] = {}
                    for j in n_adjacent_plus[i]:
                        d_t[t2][i][j] = 0  # demand carried forward at 0 is 0
            # x_tij[tt] = {i: {j: 0 for j in n_adjacent_plus[i]} for i in range(N)}

        # build UB model
        model_name = 'main_model_t%d' % tt
        mdl = gp.Model(name=model_name)
        mdl.setParam('MIPGap', 0.02)
        mdl.setParam('LogToConsole', 0)
        # Model variables
        mdl_xtij, mdl_mti, mdl_utij, mdl_dtij, mdl_theta = declare_variables(lookahead)
        print('declared variables')

        # Objective function
        objective_function(lookahead)
        print('declared objective')

        # Add constraints in a separate function
        add_constraints(lookahead)

        print('declared constraints')
        
        mdl.write('out.lp') 

        # solve model
        mdl.optimize()
        z_ub = mdl.ObjVal
        print('optimized z_ub: ', z_ub)
        
        x_t, u_tij, d_t, t_matches, m_ti_transitioning = update_variables(d_t, m_ti, D) 

        x_tot += t_matches

        if model_type in ['UB', 'heatmap']:
            m_unmatched = [sum(sum([u_tij[i][j][r] for r in range(num_drivers)]) for j in u_tij[i]) for i in u_tij]

        if enter_exit and max(extended_network) == N:
            m_ti = copy.copy(m_ti_transitioning[tt + 1]) + [0]  # 1:N
        else:
            m_ti = copy.copy(m_ti_transitioning[tt + 1])   # 1:N

        #print('\nm_ti: ', m_ti)

        if model_type == 'UB':
            u_tij_in = [[0 for r in range(num_drivers)] for _ in extended_network]
            for i in extended_network:
                for j in n_adjacent_m_in[i]:
                    for r in range(num_drivers): 
                        u_tij_in[i][r] += u_tij[j][i][r]
            for i in range(N): 
                m_ti[i] = np.add(m_ti[i], u_tij_in[i])
            sol[tt] = [str(tt) + '&' + str(sol[tt, 'd_t']) + '&' + str(D[tt]) + '&' + str(m_ti) + '&' + str(x_t)]

    sol['x_tot'] = x_tot
    elapsed_time = time.perf_counter() - t1
    sol['time'] = elapsed_time
    return sol, Demand
# Parameters and variables
# n_adjacent: neighboring nodes for repositioning
# n_adjacent_plus: nodes with positive demand
# tau_ij: travel time


def solve_sequentially(inst_string, inst_list, beta, model_type, synthetic, data_name, enter_exit,
                       disturbance, penalty_type='TAD', scenarios=10, forecast_t=5, cnt_m=0, p_dist=0.1, MHSP_gap=False):
    if synthetic:
        data = np.load(data_name, allow_pickle=True)
        dist_matrix = data['dist_matrix_ee'].tolist()
    else:
        data = np.load('data/chicago-inst-N9' + inst_string + '.npz', allow_pickle=True)
        pickle_in = open('data/dist_matrix_ee.pickle', 'rb')
        dist_matrix = pickle.load(pickle_in)

    # n_adjacent: repositioning, n_adjacent_plus: demand nodes  # data['N'].tolist()
    N, n_adjacent, n_adjacent_plus = data['N'].tolist(), data['n_adjacent'].tolist(), data['n_adjacent_plus'].tolist()
    # temp
    # N, n_adjacent_initial, n_adjacent_plus_initial = 10, data['n_adjacent'].tolist(), data['n_adjacent_plus'].tolist()
    #
    # n_adjacent = {i: [n_adjacent_initial[i][j] for j in range(len(n_adjacent_initial[i]))
    #                   if n_adjacent_initial[i][j] < 10] for i in range(N)}
    # n_adjacent_plus = {i: [n_adjacent_plus_initial[i][j] for j in range(len(n_adjacent_plus_initial[i]))
    #                        if n_adjacent_plus_initial[i][j] < 10] for i in range(N)}

    tau_ij = data['tau_ij'].tolist()
    demand_samples, T, C, S = data['demand_samples'].tolist(), data['T'].tolist(), data['C'].tolist(), 50  #  for stochastic repositioning between nodes
    m_0_vect, avg_d = data['m_0_vect_random'].tolist(), data['avg_d'].tolist()

    m_0_vect = [[2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2], [1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2], [1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2], [1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2], [1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1], [2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2]]
    
    for j in range(len(m_0_vect)): 

        num_drivers = sum(m_0_vect[j])

        m_0_vect_j = [[0 for r in range(num_drivers)] for i in range(N)]
        
        for i in range(N):
            for r in range(sum(m_0_vect[j][:i]), sum(m_0_vect[j][:i+1])): 
                m_0_vect_j[i][r] = 1

        m_0_vect[j] = m_0_vect_j

    gamma, alpha = 3, -2  # gamma: no. of heat levels, alpha: MNL distance parameter

    def compute_D_s():
        # check if I can eliminate this or introduce it somewhere else  # create sample demand scenarios
        for s in range(scenarios):
            D_s[s] = {}
            for tt in range(T):
                D_s[s][tt] = {}
                for i in range(N):
                    D_s[s][tt][i] = {}
                    for j in n_adjacent_plus[i]:
                        sample = np.random.randint(1000)
                        D_s[s][tt][i][j] = demand_samples[i][j][sample]
        return D_s
 
    for m_0 in m_0_vect[cnt_m:]:  # todo: remove cnt_m+1 (temp 09/23/22 heatmap dist)
        if cnt_m in [3]:
            cnt_m += 1
            continue
        for inst in inst_list:
            if synthetic:
                name = data_name[:-7] + '_inst_%d_v4.npz' % inst
                # name = data_name[:-4] + '_inst_%d.npz' % inst
            # else:
            #     name = 'data/chicago-inst-N9'+inst_string+'_inst_%d.npz' % inst
            demand_data = np.load(name, allow_pickle=True)
            D = demand_data['D'].tolist()
            if synthetic:
                D_s = demand_data['D_s'].tolist()
            else:
                D_s = compute_D_s()
            sol, Demand = build_model(m_0, N, n_adjacent, n_adjacent_plus, S, D, D_s, scenarios, T,
                                    tau_ij, avg_d, gamma, alpha, dist_matrix, beta, model_type, enter_exit, disturbance,
                                      p_dist, forecast_t, MHSP_gap, penalty_type)
            print(str(sol['x_tot']) + " out of " + str(Demand) + ' demand')
            sol['m0', tuple(m_0)] = str(sol['x_tot']) + " out of " + str(Demand) + ' demand'
            namesol = name[:-4] + '_sol_same_zone_' + inst_string + '_' + str(cnt_m) + 'S_%d_lookahead_%d_' % (scenarios, forecast_t) \
            + 'HIP_' + penalty_type  # + '_v2'
            np.save(namesol, sol)
        cnt_m += 1
