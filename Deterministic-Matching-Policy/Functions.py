"""
Fleet Size Planning with Wage Guarantee Constraints

This module implements the core algorithms from:
    "Fleet Size Planning in Crowdsourced Delivery: 
     Balancing Service Level and Driver Utilization"

THE RESEARCH PROBLEM:
    Crowdsourced delivery platforms face a fundamental trade-off:
    - Too many drivers → High demand fulfillment, but drivers are underutilized
    - Too few drivers → Drivers are busy, but orders go unfulfilled
    
    The platform must decide: How many drivers to activate each period?
    
    Added complexity: Platforms often guarantee drivers a minimum wage 
    (e.g., $15/hour), which they achieve if utilization > threshold W (typically 80%).

KEY DECISIONS:
    1. Pool Size x[p]: How many drivers to invite/activate in period p
    2. Matching x[a,b]: Which driver a serves which order b

KEY METRICS:
    1. Service Level: Fraction of demand (orders) fulfilled
    2. Driver Utilization Rate: (Time spent delivering) / (Total active time)
    3. Wage Guarantee Rate: Fraction of drivers meeting utilization threshold W

ALGORITHMS IMPLEMENTED:

    1. Softmax (Boltzmann) Policy for Pool Sizing:
       - Learn optimal pool sizes through value function V[period, size]
       - Temperature annealing for exploration-exploitation balance
       - See function: boltzmann()

    2. Utilization-Aware Matching:
       - LP assigns drivers to orders in real-time
       - Priority score ρ_a gives preference to underutilized drivers
       - Helps drivers meet wage guarantee threshold
       - See function: solve_opt(), cal_rho_a()

    3. Stochastic Simulation:
       - Poisson demand arrivals with rate λ
       - Binomial driver arrivals from pool
       - See function: sample_path()

DATA STRUCTURES:
    - Driver: Tracks location, utilization, earnings, availability
    - Demand: Tracks origin/destination, deadline, value
    - Value function V[(period, pool_size)] → expected profit

Dependencies:
    - gurobipy: LP solver for matching optimization
    - numpy, scipy: Numerical computing and distance calculations
    
Author: Sahil Bhatt
"""

import numpy as np
import math
from scipy.spatial import distance
from gurobipy import *
from scipy.spatial.distance import euclidean
import copy
import random
from typing import Dict, List, Tuple, Any


def boltzmann(V: Dict, iteration: int) -> List[int]:
    """
    Softmax (Boltzmann) policy for pool size selection.
    
    This is the KEY DECISION mechanism for fleet size planning.
    Instead of using a fixed pool size, we learn which pool sizes
    work best for each planning period through the value function V.
    
    The Policy:
        P(pool_size = k | period = j) ∝ exp(-ζ * V[j,k])
        
    where:
        - V[j,k] = expected cost/regret of choosing pool size k in period j
        - ζ (zeta) = temperature parameter controlling exploration
        - Lower V[j,k] → higher probability of selecting k
    
    Temperature Annealing:
        - Early iterations: High temperature → explore many pool sizes
        - Later iterations: Low temperature → exploit best-performing sizes
        - Formula: ζ = iteration / (10 * value_range)
    
    Why Softmax over ε-greedy?
        - Smoother exploration across action space
        - Naturally handles continuous-like pool size ranges
        - Better convergence properties for this problem
    
    Args:
        V: Value function dictionary V[period, pool_size] → expected cost
           Learned through simulation iterations
        iteration: Current training iteration (controls temperature)
        
    Returns:
        List of 16 pool sizes, one for each planning period
        Values range from 1 to 55 drivers
        
    Example:
        >>> V = {(p, k): some_cost for p in range(16) for k in range(1, 56)}
        >>> pool_sizes = boltzmann(V, iteration=100)
        >>> print(pool_sizes)  # e.g., [25, 30, 35, 40, 42, 38, ...]
    """
    num_periods = 16
    x_hat_star = [0] * num_periods
    upper_limit = 55  # Maximum pool size
    
    for j in range(num_periods):
        # Calculate temperature based on value function range
        v_max = max(V[(j, k)] for k in range(1, upper_limit + 1))
        v_min = min(V[(j, k)] for k in range(1, upper_limit + 1))
        d = v_max + 0.0001 - v_min  # Value range (avoid division by zero)
        
        # Temperature decreases with iteration (annealing schedule)
        zeta = iteration / (10 * d)
        
        # Compute Boltzmann weights
        W = 0
        w = [0] * upper_limit
        
        for k in range(1, upper_limit + 1):
            w[k - 1] = math.exp(-zeta * V[(j, k)])
            W += w[k - 1]
        
        # Sample action proportional to weights
        u = random.random() * W
        
        wtemp = 0
        k = 0
        while True:
            wtemp += w[k]
            if wtemp >= u:
                break
            k += 1
            
        x_hat_star[j] = 1 + k
        
    return x_hat_star


def closest_node(node: List[float], loc_set: List[List[float]]) -> List[float]:
    """Map continuous coordinates to nearest discrete grid point."""
    nodes = np.asarray(loc_set)
    closest_ind = distance.cdist([node], nodes).argmin()
    return loc_set[closest_ind]


def sample_path(mu_enter: float, mu_exit: float, lambd: float,
                grid_size: Tuple[int, int], t_interval: float, T: int,
                loc_set: List, num_drivers_enter: List[int]) -> Tuple[Dict, Dict]:
    """
    Generate a sample path (trajectory) for one simulation iteration.
    
    Simulates the arrival of demand (orders) and drivers over T time epochs
    following Poisson processes. This creates a stochastic scenario for
    evaluating workforce scheduling policies.
    
    Args:
        mu_enter: Mean driver arrival rate (not used - controlled externally)
        mu_exit: Mean driver departure rate (not used here)
        lambd: Mean demand arrival rate per epoch (Poisson parameter)
        grid_size: (width, height) of the service region
        t_interval: Duration of each time epoch (minutes)
        T: Total number of epochs in planning horizon
        loc_set: List of valid discrete locations
        num_drivers_enter: Pre-specified driver arrivals per epoch
        
    Returns:
        Tuple of (demand_dict, driver_dict) containing sample path data
    """
    m_n = {}  # Driver sample path
    d_n = {}  # Demand sample path
    
    for t in range(T):
        # Driver arrivals (controlled by pool sizing decisions)
        m_t = num_drivers_enter[t]
        m_n[t] = m_t

        if m_t > 0:
            for i in range(m_t):
                # Random location on grid
                m_n[t, i] = closest_node(
                    [np.random.uniform(0, grid_size[0]),
                     np.random.uniform(0, grid_size[1])],
                    loc_set
                )

        # Demand arrivals (Poisson process)
        d_t = np.random.poisson(lambd)
        d_n[t] = d_t
        
        if d_t > 0:
            for i in range(d_t):
                # Generate origin-destination pair
                origin = closest_node(
                    [np.random.uniform(0, grid_size[0]),
                     np.random.uniform(0, grid_size[1])],
                    loc_set
                )
                destination = closest_node(
                    [np.random.uniform(0, grid_size[0]),
                     np.random.uniform(0, grid_size[1])],
                    loc_set
                )
                
                while origin == destination:
                    destination = closest_node(
                        [np.random.uniform(0, grid_size[0]),
                         np.random.uniform(0, grid_size[1])],
                        loc_set
                    )
                
                # Order value ~ U[3, 12] representing delivery fee
                value_d = np.random.uniform(3, 12)
                announce_time = np.random.uniform(
                    max(0, (t - 1)) * t_interval,
                    t * t_interval
                )
                
                d_n[t, i] = [[origin, destination], announce_time, value_d]

    return d_n, m_n


class Driver:
    """
    Gig worker (driver) entity for workforce scheduling.
    
    Tracks driver state including availability, location, utilization,
    and earnings for enforcing minimum wage guarantees.
    
    Attributes:
        number: Unique driver identifier
        ma: Availability status (1=available for matching)
        loc: Current [x, y] location
        time_entered: When driver joined the platform
        exit_time: Planned departure time
        ha: Cumulative active/working time
        bar_ha: Utilization rate = working_time / total_time
        profit_matching: Cumulative earnings from deliveries
    """
    
    def __init__(self, number: int, ma: int, loc: List[float],
                 time_entered: float, exit_time: float,
                 ha: float, bar_ha: float, profit_matching: float):
        self.number = number
        self.ma = ma
        self.loc = loc
        self.ha = ha
        self.bar_ha = bar_ha
        self.profit_matching = profit_matching
        self.active_history = []
        self.iteration_history = {}
        self.time_entered = time_entered
        self.exit_time = min(exit_time, 192)  # Cap at planning horizon

    def __str__(self):
        return f'Driver {self.number}: utilization={self.bar_ha:.2%}'


class Demand:
    """
    Delivery order entity with time window constraints.
    
    Attributes:
        number: Unique order identifier
        origin: Pickup location
        destination: Delivery location
        dst: Euclidean distance (origin to destination)
        announce_time: When order was placed
        placement_time: Epoch when order entered system
        delivery_deadline: Latest acceptable delivery time
        value: Revenue for completing this order
    """
    
    def __init__(self, number: int, origin: List[float], destination: List[float],
                 dist: float, announce_time: float, placement_time: float,
                 delivery_deadline: float, value: float):
        self.number = number
        self.origin = origin
        self.destination = destination
        self.dst = dist
        self.announce_time = announce_time
        self.delivery_deadline = delivery_deadline
        self.value = value
        self.iteration_history = {}
        self.placement_time = placement_time

    def __str__(self):
        return f'Order {self.number}: ${self.value:.2f}'


def cal_rho_a(bar_ha: float) -> float:
    """
    Calculate driver priority score based on utilization.
    
    Priority function incentivizes matching underutilized drivers
    to help them meet minimum guarantee thresholds (W = 0.8).
    
    Args:
        bar_ha: Current driver utilization ∈ [0, 1]
        
    Returns:
        Priority score ρ_a ∈ [0, 1]
    """
    if bar_ha < 0.8:
        return -1.25 * bar_ha + 1
    return 0


def cal_rho_b(perc: float) -> float:
    """
    Calculate demand priority based on deadline urgency.
    
    Args:
        perc: Fraction of time remaining until deadline
        
    Returns:
        Priority score ρ_b ∈ [0, 0.5]
    """
    if perc <= 0.25:
        return 0.5
    elif perc > 0.5:
        return 0
    return -perc + 0.5


def cal_theta(dist: float) -> float:
    """
    Distance-based delivery fee scaling.
    
    Args:
        dist: Delivery distance
        
    Returns:
        Fee rate multiplier
    """
    if dist < 2:
        return 4
    elif dist < 5:
        return 6
    return 7
    
def solve_opt(data, t, n, demand_list, driver_list, driver_attr, demand_attr, penalty):
    model_name = 'LP_t%d_n%d' % (t, n)
    mdl = Model(model_name)
    mdl.params.LogToConsole = 0 
    # define decision variables: x_tab
    x_ab_feasible = {('b', b): [] for b in demand_list['active']}
    x_ab_cost, x_ab_time, x_ab_dist, keys = {}, {}, {}, []

    drivers = driver_list['available'] + driver_list['unavailable']
    for a in drivers:
        x_ab_feasible[('a', a)] = [0]
        keys.append((a, 0)) 
        
        tot_time = (t+1-driver_attr[a].time_entered)*data['t_interval'] 
        
        #instead of discretizing the bar_ha I used the exact quantity for more consistency and less oscillation 
        
        bar_ha_new = round(driver_attr[a].ha / tot_time * 100 / data['h_int']) * data['h_int'] / 100

        bar_ha_new = driver_attr[a].ha / tot_time

        # obj func value when unmatched (matched to 0)
        rho_a = cal_rho_a(bar_ha_new)
        x_ab_cost[(a, 0)] = (-penalty * rho_a)*data['w_d']
        x_ab_cost[(a, 0), 'driver'] = [0, 0]
        x_ab_cost[(a, 0), 'platform_profit'] = 0
         
        a_loc = tuple(driver_attr[a].loc)

        for b in demand_list['active']:
            b_loc_o = tuple(demand_attr[b].origin)
            dst = distance.euclidean(a_loc, b_loc_o)
            dst_o_d = demand_attr[b].dst
            dst_tot = dst + dst_o_d  # dst from driver to origin + package orig to dest.
            x_ab_dist[(a, b)] = dst
            x_ab_dist[(a, b, 'tot_dst')] = dst_tot
            cont_time = data['eta'] * dst_tot
            disc_time = math.ceil(cont_time / data['t_interval']) * data['t_interval']
            time = disc_time + (t * data['t_interval'])  # time interval + time of t
            t_epoch = int(time/data['t_interval'])
 
            if time <= demand_attr[b].delivery_deadline: #the exiting drivers are already eliminated in predecisiosn; do not check if the driver exits
                #drivers can be assigned before exiting even if they would have to exit at a later time 
                # find aggregate expected location:
                loc = demand_attr[b].destination  # loc of demand dest 
                # 0.6/mile = 0.3728/km = 1.1185/unit (1 unit = 3 km) + $0.2/min
                fixed_charge = 2.5  # fixed charge per pickup and delivery
                driver_pay_ab = (1.1185 * dst_tot + 0.2 * disc_time + fixed_charge)
                driver_profit = 0.75 * driver_pay_ab - (0.5 * 1.1185 * dst_tot)  # pay minus per km cost

                tot_time = (t+1-driver_attr[a].time_entered)*data['t_interval'] 
                 
                bar_ha_new = round((driver_attr[a].ha + disc_time) / tot_time * 100 / data['h_int']) * data['h_int'] / 100
                
                #instead of discretizing the bar_ha I used the exact quantity for more consistency and less oscillation 

                bar_ha_new = (driver_attr[a].ha + disc_time) / tot_time
                
                keys.append((a, b))
                x_ab_feasible[('a', a)].append(b)
                x_ab_feasible[('b', b)].append(a)

                # removed since not parametric cost function
                perc = 1 - float(time / demand_attr[b].delivery_deadline)  # perc of time window remaining
                rho_b = cal_rho_b(perc)

                rho_a = cal_rho_a(bar_ha_new)  # get rho_a for each a/b match
                theta = cal_theta(dst_o_d) #NOTE: using a piecewise linear function of dist
                phi = demand_attr[b].value

                c =  data['w_s']*(theta + phi - 0.5*driver_pay_ab) + data['alpha_penalty'] * (data['w_d']*rho_a + data['w_s']*rho_b)\
                        + data['w_d']*(-penalty * rho_a)
                
                x_ab_cost[(a, b), 'driver'] = [driver_profit, 0]
                x_ab_cost[(a, b), 'platform_profit'] = theta + phi - 0.5*driver_pay_ab
                # under this scenario, driver only earns profit from matching. Paid by platform = 0.

                # Note: right now only accounting for distance from driver to origin 
                x_ab_cost[(a, b)] = c 
                x_ab_time[(a, b)] = time
                
    # define decision variables: x_tab; commented out cplex version
    #mdl.xab = mdl.continuous_var_dict(keys, lb=0)
 
    xab = {key: mdl.addVar(lb=0, name=f'x_{key[0]}_{key[1]}', vtype = GRB.CONTINUOUS) for key in keys}
 
    # define cost function; commented out cplex version 
    # mdl.match = mdl.sum(mdl.xab[(a, b)] * x_ab_cost[(a, b)] for a in drivers
    #                     for b in x_ab_feasible[('a', a)])
    
    #mdl.maximize(mdl.match)

    mdl.setObjective(quicksum((xab[(a, b)] * x_ab_cost[(a, b)] for a in drivers for b in x_ab_feasible[('a', a)])), GRB.MAXIMIZE)
     
    # Add constraints in a separate function
    # resource constraint 

    a_cnst = mdl.addConstrs(quicksum(xab[(a, b)] for b in x_ab_feasible[('a', a)]) == 1 for a in drivers)

    # demand constraint 
    
    mdl.addConstrs(quicksum(xab[(a, b)] for a in x_ab_feasible[('b', b)]) <= 1 for b in demand_list['active'])
     
    mdl.optimize()
    # Get obj and var values
    z = mdl.ObjVal

    #print('z = ' + str(z))
    xab = {('b', 0): []} #different from x_a_b variables of the model defined in setup_model_cfa
    
    #print('a_cnst: ', a_cnst); print('len a_cnst: ', len(a_cnst));  

    dst, tot_dst, platform_profit = 0, 0, 0
    # Get solution values
    num_matches = 0
    drivers = driver_list['available'] 
     
    # print('t: ', t); print() 

    # print('drivers: ', drivers); print('len(drivers): ', len(drivers)) 
    # print() 
    # print('driver bar_ha: ', [driver_attr[drivers[i]].bar_ha for i in range(len(drivers))])

    for i in range(len(drivers)):
        a = drivers[i]
        for b in x_ab_feasible[('a', a)]: 
            if mdl.getVarByName(f"x_{a}_{b}").X >= 0.99: #changed from cplex version: if mdl.xab[(a, b)].solution_value >= 0.99:     
                xab[('a', a)] = b
                if b != 0:
                    dst += x_ab_dist[(a, b)]
                    tot_dst += x_ab_dist[(a, b, 'tot_dst')]
                platform_profit += x_ab_cost[(a, b), 'platform_profit']
                break  # break out of the b loop as soon as you find match
        if xab[('a', a)] != 0:
            xab[('b', b)] = a
            num_matches += 1
        else:
            xab[('b', 0)].append(a)  # driver unmatched 

    #print('xab: ', xab)

    #print(['a: '+str(a)+' b: '+str(b) + ' cost: ' + str(x_ab_cost[(a, b)]) for a in drivers for b in x_ab_feasible[('a', a)]]) 

    #print()

    return xab, z, num_matches, x_ab_time, x_ab_cost, dst, tot_dst, platform_profit
 
# Step 3.4a: # Done
# This func updates driver and demand attributes after matching.
# Verified for 1. disagg alg, 2. agg alg, 3. Test_VFA_convergence_N
def update_attr(driver_list, demand_list, xab, driver_attr, demand_attr, t, data, x_ab_time, x_ab_cost, stats=False,
                driver_log=[], driver_stats=[], demand_stats=[]):
    # data = {'t_interval': [], 'h_int': [], 'Guaranteed_service': [], 'gamma': [], 'eta': [], 'W': [], 'T': []}
    driver_temp = {'av': [], 'unav': [], 'exit': []}
    drivers = driver_list['available']  
    #print('driver_stats: ', driver_stats)

    for a in drivers:
        # if driver was not matched at t
        if xab[('a', a)] == 0: 
            if stats:
                if a not in driver_log:
                    driver_log[a] = {'all_t':[]}
                elif 'all_t' not in driver_log[a]:
                    driver_log[a]['all_t'] = []
                driver_log[a]['all_t'].append(['t%d_post-decision' % t, 'unmatched', 'active',
                                                [driver_attr[a].ha, driver_attr[a].bar_ha]])

            if t >= driver_attr[a].exit_time:
                bar_ha = driver_attr[a].bar_ha
                driver_attr[a].active_history.append(0)
                profit_matching = driver_attr[a].profit_matching
                if bar_ha >= data['W']:
                    driver_stats['guarantee_met'].append([a, bar_ha, [profit_matching],
                                                            driver_attr[a].active_history])
                else:
                    driver_stats['guarantee_missed'].append([a, bar_ha, [profit_matching],
                                                            driver_attr[a].active_history]) 
                driver_temp['exit'].append(a)
            else:  # duration < guaranteed service (& driver was not matched)
                busy_time = driver_attr[a].ha
                tot_time = (t+1-driver_attr[a].time_entered)*data['t_interval'] 
                 
                # total prev time(active time/perc active) + t interval
                bar_ha = round(busy_time/tot_time*100/data['h_int'])*data['h_int']/100

                bar_ha = busy_time/tot_time

                if bar_ha > 1.0:
                    print('Stop and debug!')
                driver_attr[a].bar_ha = bar_ha
                driver_attr[a].profit_matching += x_ab_cost[(a, 0), 'driver'][0]
                driver_attr[a].active_history.append(0)
                driver_temp['av'].append(a)
                driver_attr[a].iteration_history[t] = "This driver was not matched at t = " + str(t) \
                                                        + " and is in location " + str(driver_attr[a].loc) + "."
        else:  # Next: continue att update if driver was matched
            b = xab[('a', a)]  
            # check if time it takes to make delivery will be before next t
            if x_ab_time[(a, b)] <= (t+1)*data['t_interval']:  # if fulfill delivery before next period
                driver_temp['av'].append(a)
                driver_attr[a].active_history.append(1)  # active for one epoch
            # if driver will still be delivering by t+1
            else:
                # unavailable, next time available is x_ab_time or possibly change entry 1 to x_ab_time right away
                driver_attr[a].ma = 0
                driver_attr[a].available_time = x_ab_time[(a, b)]
                # num epochs to fulfill order
                num_epochs = int((x_ab_time[(a, b)] - (t * data['t_interval'])) / data['t_interval'])
                driver_attr[a].active_history += [1 for _ in range(num_epochs)]
                driver_temp['unav'].append(a)

            driver_attr[a].profit_matching += x_ab_cost[(a, b), 'driver'][0]
 
            busy_time = driver_attr[a].bar_ha*driver_attr[a].ha + (x_ab_time[(a, b)] - t * data['t_interval'])  # additional busy time t

            tot_time = x_ab_time[(a, b)]-driver_attr[a].time_entered*data['t_interval']
 
            if tot_time < 1e-6:
                print('Why is this zero?')  # loc = demand
                perc = 0.0
            else:
                perc = float(busy_time / tot_time)
            driver_attr[a].iteration_history[t] = \
                "This driver was matched with order %d at time t = %d, and is in location %s at time %f." % (b, t,
                                                            str(driver_attr[a].loc),x_ab_time[(a, b)])
            loc = demand_attr[b].destination
            
            driver_attr[a].loc = loc 
            driver_attr[a].ha = busy_time
            bar_ha = round(perc*100/data['h_int'])*data['h_int']/100

            bar_ha = perc
            if bar_ha > 1.0:
                print('Stop and debug!')
            driver_attr[a].bar_ha = bar_ha

            if stats:
                if a not in driver_log:
                    driver_log[a] = {'all_t':[]}
                elif 'all_t' not in driver_log[a]:
                    driver_log[a]['all_t'] = [] 
                driver_log[a]['all_t'].append(['t%d_post_decision' % t,
                    [demand_attr[b].origin, demand_attr[b].destination, x_ab_time[(a, b)], busy_time,
                     driver_attr[a].bar_ha]])

    if t > 0:  # check if previously unavailable drivers are now available
        if len(driver_list['unavailable']) > 0:
            for a in driver_list['unavailable']: 
                avail_time = driver_attr[a].available_time
                if avail_time <= (t+1)*data['t_interval']:  # if driver will be avail at t+1
                    if t >= driver_attr[a].exit_time: 
                        # driver attr resets if he meets the x hours threshold
                        loc = driver_attr[a].loc 

                        if stats:
                            bar_ha = driver_attr[a].bar_ha
                            profit_matching = driver_attr[a].profit_matching
                            driver_attr[a].active_history.append(0)  # 0 for unmatched, 1 for matched

                            if bar_ha >= data['W']:
                                driver_stats['guarantee_met'].append([a, bar_ha, [profit_matching], driver_attr[a].active_history])
                            else:
                                driver_stats['guarantee_missed'].append([a, bar_ha, [profit_matching], driver_attr[a].active_history])
                    
                        driver_temp['exit'].append(a)
                    else:  # done deliv, within guaranteed service  --> available, active
                        driver_attr[a].ma = 1
                        driver_temp['av'].append(a) 
                else:  # if not done delivering, move it to unavailable t+1
                    driver_temp['unav'].append(a) 

    #print('t: ', t, ' driver_stats: ', driver_stats, ' stats: ', stats)
    if stats: 
        driver_stats[t, 'active'] = len(driver_temp['av']) + len(driver_temp['unav'])

    demand_temp = {"active": []}  # Now, move on to update demand info :)
    for b in demand_list['active']:
        if ('b', b) not in xab:  # if demand b is not satisfied carry it forward if not missed
            # if deadline is after t+1 + min time
            if demand_attr[b].delivery_deadline >= ((t+1)*data['t_interval']) + (data['eta']*demand_attr[b].dst):
                demand_temp['active'].append(b)
            else:
                demand_list['expired'].append(b)
                if stats:
                    demand_stats['missed'].append(b)
        else:
            demand_list['fulfilled'].append(b)
            if stats: 
                demand_stats['met'].append(b)
    return driver_temp, demand_temp, driver_stats, demand_stats, driver_attr, demand_attr


# Verified for 1. disagg alg, 2. agg alg, 3. Test_VFA_convergence_N
def predecision_state(num_drivers, num_demand, driver_temp, demand_temp, driver_list, demand_list, t, n, m, d,
                      driver_attr, demand_attr, data, train, stats=False, driver_log=[], driver_stats=[]):
    # (1) check if any drivers are unmatched, if yes, one random leaves if mu_exit>0
    # (2) new arrivals of both drivers and demand from sample path.
    # Note: create complete sample path of n=1 in data generation
    #print('pred driver_stats: ', driver_stats)
    
    if train:  # train = True if training model
        m_n, d_n = m[n], d[n]
    else:
        m_n, d_n = m, d 

    def update_driver_attr(num_drivers):
        driver_list['available'] = copy.copy(driver_temp['av'])
        driver_list['unavailable'] = copy.copy(driver_temp['unav'])
        driver_list['exit'] = copy.copy(driver_temp['exit'])

        # Let go of drivers who reached guarantee
        for a in driver_list['available']:
            if t >= driver_attr[a].exit_time:
                loc = driver_attr[a].loc 

                if stats:
                    bar_ha = driver_attr[a].bar_ha
                    profit_matching = driver_attr[a].profit_matching
                    if bar_ha >= data['W']:
                        driver_stats['guarantee_met'].append([a, bar_ha, [profit_matching],
                                                              driver_attr[a].active_history])
                    else:
                        driver_stats['guarantee_missed'].append([a, bar_ha, [profit_matching],
                                                                 driver_attr[a].active_history])
 
                driver_list['exit'].append(a)
                driver_list['available'].remove(a)
                if stats:
                    driver_log[a]['all_t'].append(['t%d' % (t+1), [0, 1, loc, 0, 0, 0, 0]])
                    # todo: maybe update to also log loc_1 and loc_2
            else:
                if stats:
                    if a not in driver_log:
                        driver_log[a] = {'all_t':[]}
                    elif 'all_t' not in driver_log[a]:
                        driver_log[a]['all_t'] = [] 

                    driver_log[a]['all_t'].append(['t%d' % (t + 1), [driver_attr[a].ma,
                     driver_attr[a].loc, driver_attr[a].time_entered, driver_attr[a].ha, driver_attr[a].bar_ha]])
 
        # new drivers
        if stats and t+1 >= data['T']:
            pass
        else:
            for j in range(m_n[t+1]):  # create new driver objects range(m[n][t+1])
                a = j+num_drivers
                loc = m_n[t+1, j] 

                if stats:
                    driver_log[a] = {'all_t': ['t%d' % (t + 1), [0, 1, loc, 0, 0, 0]]}
                driver_attr[a] = Driver(a, 1, loc, t+1, t+1+1/5 * np.random.uniform(30, 90), 0, 0, 0)

                driver_list['available'].append(a)
            num_drivers += m_n[t+1]
        return num_drivers

    def update_demand_attr(num_demand):
        demand_list['active'] = copy.copy(demand_temp['active'])

        if stats and t+1 >= data['T']:
            pass
        else:
            # new demand
            for j in range(d_n[t+1]):  # no. of new demands
                b = num_demand + j + 1  # unique identifier
                i_loc_o = d_n[t+1, j][0][0]
                i_loc_d = d_n[t+1, j][0][1]
                dst = distance.euclidean(i_loc_o, i_loc_d)
                announce_time = d_n[t+1, j][1]
                value_d = d_n[t+1, j][2]
                demand_attr[b] = Demand(b, i_loc_o, i_loc_d, dst, announce_time, t+1, announce_time + data['delta_t'], value_d)
                demand_list['active'].append(b)
            num_demand += d_n[t+1]
        return num_demand

    num_drivers = update_driver_attr(num_drivers)
    num_demand = update_demand_attr(num_demand)
    return num_drivers, num_demand, driver_stats, driver_attr, demand_attr





