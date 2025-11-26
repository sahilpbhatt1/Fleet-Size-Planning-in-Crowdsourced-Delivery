import numpy as np
import math
from scipy.spatial import distance
from docplex.mp.model import Model
import copy
import random

# temporary file for modifications to functions file to include possibility of aggregation
# data =
# {'t_interval': [], 'h_int': [], 'Guaranteed_service': [], 'gamma': [], 'eta': [], 'W': [], 'T': [], 'delta_t': []}
# Find closest node from discrete set

def closest_node(node, loc_set):
    nodes = np.asarray(loc_set)
    closest_ind = distance.cdist([node], nodes).argmin()
    return loc_set[closest_ind]


# Driver and demand classes to generate objects
# Verified for 1. disagg alg, 2.
class Driver:
    def __init__(self, number, sa, ma, loc, time_first_matched, ha, bar_ha):
        self.number = number
        self.sa = sa    # binary=1 if driver active
        self.ma = ma    # binary=1 if available for matching
        self.loc = loc  # o_a current location
        self.time_first_matched = time_first_matched
        self.ha = ha    # active time spent up to t
        self.bar_ha = bar_ha    # portion of active time driver is utilized
        self.iteration_history = {}

    def __str__(self):
        return 'This is driver no. {self.number}'.format(self=self)


# Verified for 1. disagg alg, 2.
class Demand:
    def __init__(self, number, origin, destination, dist, announce_time, delivery_deadline):
        self.number = number
        self.origin = origin
        self.destination = destination
        self.dst = dist
        self.announce_time = announce_time
        self.delivery_deadline = delivery_deadline
        self.iteration_history = {}

    def __str__(self):
        return 'This is demand no. {self.number}'.format(self=self)


# compute rho_a (priority score)
def cal_rho_a(bar_ha):
    # rho_a = [[0, 0.8, 1], [1, 0, 0]]
    # rho_b = [[[0, 0.24999999], [0.5, 0.5]], [[0.25, 0.5, 1], [0.25, 0, 0]]]
    if bar_ha < 0.8:
        rho_a = -1.25 * bar_ha + 1  # eqn of line from above
        # rho_a = -0.625 * bar_ha + 1  # changed line to be 0.5 penalty at 0.8
    # elif bar_ha <= 0.9:  # changed 02/09/21
    #     rho_a = -5*bar_ha + 4.5
    else:
        rho_a = 0
    return rho_a


# compute rho_b (priority score)
def cal_rho_b(perc):  # perc away from min time window to fulfill order
    if perc <= 0.25:
        rho_b = 0.5
    elif perc > 0.5:
        rho_b = 0
    else:
        rho_b = -1 * perc + 0.5  # eqn of line from above points
    return rho_b


# Setup model
def setup_model(data, t, demand_list, driver_list, driver_attr, demand_attr, penalty):
    model_name = 'LP_t%d' % t
    mdl = Model(name=model_name)
    # define decision variables: x_tab
    x_ab_feasible = {('b', b): [] for b in demand_list['active']}
    x_ab_cost, x_ab_time, x_ab_dist, keys = {}, {}, {}, []

    drivers = driver_list['act_available'] + driver_list['inactive']
    for a in drivers:
        x_ab_feasible[('a', a)] = [0]
        keys.append((a, 0))
        sa = driver_attr[a].sa
        if sa == 1:
            tot_time = (driver_attr[a].ha / driver_attr[a].bar_ha) + data['t_interval']
            g = max(data['Guaranteed_service'] - ((t + 1) * data['t_interval'] - driver_attr[a].time_first_matched), 0)
            bar_ha_new = round(driver_attr[a].ha / tot_time * 100 / data['h_int']) * data['h_int'] / 100

        # obj func value when unmatched (matched to 0)
        if sa == 0 or (sa == 1 and g > 0) or (sa == 1 and g <= 0 and bar_ha_new >= data['W']):
            x_ab_cost[(a, 0)] = 0
        else:
            rho_a = cal_rho_a(driver_attr[a].bar_ha)
            x_ab_cost[(a, 0)] = -penalty * rho_a  # penalty = -1000 for high penalty or -1 for regular

        a_loc = tuple(driver_attr[a].loc)
        for b in demand_list['active']:
            b_loc_o = tuple(demand_attr[b].origin)
            dst = distance.euclidean(a_loc, b_loc_o)
            dst_o_d = demand_attr[b].dst
            dst_tot = dst + dst_o_d  # dst from driver to origin + package orig to dest.
            cont_time = data['eta'] * dst_tot
            disc_time = math.ceil(cont_time / data['t_interval']) * data['t_interval']
            time = disc_time + (t * data['t_interval'])  # time interval + time of t
            x_ab_dist[(a, b)] = dst
            x_ab_dist[(a, b, 'tot_dst')] = dst_tot
            t_epoch = int(time/data['t_interval'])

            if time <= demand_attr[b].delivery_deadline:  # if can be delivered before deadline
                sa = 1
                if driver_attr[a].sa == 0:
                    bar_ha_new = 1.0
                    g = max(data['Guaranteed_service'] - disc_time, 0)
                else:
                    if driver_attr[a].bar_ha == 0:
                        print('Not sure why bar_ha = %d, ha = %d, sa = %d' % (driver_attr[a].bar_ha,
                                                                              driver_attr[a].ha, sa))
                        tot_time = disc_time
                    else:
                        tot_time = (driver_attr[a].ha / driver_attr[a].bar_ha) + disc_time
                    bar_ha_new = round((driver_attr[a].ha + disc_time) / tot_time * 100 / data['h_int']) * data['h_int'] / 100
                    g = max(data['Guaranteed_service'] - (time - driver_attr[a].time_first_matched), 0)

                keys.append((a, b))
                x_ab_feasible[('a', a)].append(b)
                x_ab_feasible[('b', b)].append(a)

                if g > 0 or (g <= 0 and bar_ha_new >= data['W']):  # if there's time or no time but guarantee met
                    c = (data['theta1'] * dst_o_d) - (data['theta'] * dst)
                else:  # no time left but guarantee unmet
                    rho_a = cal_rho_a(bar_ha_new)
                    c = (data['theta1'] * dst_o_d) - (data['theta'] * dst) + (-penalty * rho_a)

                # Note: right now only accounting for distance from driver to origin
                x_ab_cost[(a, b)] = c
                x_ab_time[(a, b)] = time
    # define decision variables: x_tab
    mdl.xab = mdl.continuous_var_dict(keys, lb=0)

    # define cost function
    mdl.match = mdl.sum(mdl.xab[(a, b)] * x_ab_cost[(a, b)] for a in drivers
                        for b in x_ab_feasible[('a', a)])
    mdl.maximize(mdl.match)

    # Add constraints in a separate function
    # resource constraint
    a_cnst = mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for b in x_ab_feasible[('a', a)]) == 1, 'Resource_a%d'
                                  % a) for a in drivers)

    # demand constraint
    mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for a in x_ab_feasible[('b', b)]) <= 1, 'Demand_b%d'
                         % b) for b in demand_list['active'])
    return mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst

def setup_model_cfa(data, t, demand_list, driver_list, driver_attr, demand_attr, penalty):
    model_name = 'LP_t%d' % t
    mdl = Model(name=model_name)
    # define decision variables: x_tab
    x_ab_feasible = {('b', b): [] for b in demand_list['active']}
    x_ab_cost, x_ab_time, x_ab_dist, keys = {}, {}, {}, []

    drivers = driver_list['act_available'] + driver_list['inactive']
    for a in drivers:
        x_ab_feasible[('a', a)] = [0]
        keys.append((a, 0))
        sa = driver_attr[a].sa
        # obj func value when unmatched (matched to 0)
        # cfa: if guarantee not met, penalize throughout activity window not just end of horizon
        if sa == 0 or (sa == 1 and driver_attr[a].bar_ha > data['W']):
            x_ab_cost[(a, 0)] = 0
        else:
            rho_a = cal_rho_a(driver_attr[a].bar_ha)
            x_ab_cost[(a, 0)] = -penalty * rho_a  # penalty = -1000 for high penalty or -1 for regular

        a_loc = tuple(driver_attr[a].loc)
        for b in demand_list['active']:
            b_loc_o = tuple(demand_attr[b].origin)
            dst = distance.euclidean(a_loc, b_loc_o)
            dst_o_d = demand_attr[b].dst
            dst_tot = dst + dst_o_d  # dst from driver to origin + package orig to dest.
            cont_time = data['eta'] * dst_tot
            disc_time = math.ceil(cont_time / data['t_interval']) * data['t_interval']
            time = disc_time + (t * data['t_interval'])  # time interval + time of t
            x_ab_dist[(a, b)] = dst
            x_ab_dist[(a, b, 'tot_dst')] = dst_tot
            t_epoch = int(time/data['t_interval'])

            if time <= demand_attr[b].delivery_deadline:  # if can be delivered before deadline
                if driver_attr[a].sa == 0:
                    bar_ha_new = 1.0
                    g = max(data['Guaranteed_service'] - disc_time, 0)
                else:
                    if driver_attr[a].bar_ha == 0:
                        print('Not sure why bar_ha = %d, ha = %d, sa = %d' % (driver_attr[a].bar_ha,
                                                                              driver_attr[a].ha, sa))
                        tot_time = disc_time
                    else:
                        tot_time = (driver_attr[a].ha / driver_attr[a].bar_ha) + disc_time
                    bar_ha_new = round((driver_attr[a].ha + disc_time) / tot_time * 100 / data['h_int']) * data['h_int'] / 100
                    g = max(data['Guaranteed_service'] - (time - driver_attr[a].time_first_matched), 0)
                keys.append((a, b))
                x_ab_feasible[('a', a)].append(b)
                x_ab_feasible[('b', b)].append(a)

                perc = 1 - float(time / demand_attr[b].delivery_deadline)  # perc of time window remaining
                rho_b = cal_rho_b(perc)

                rho_a = cal_rho_a(bar_ha_new)  # get rho_a for each a/b match
                if g > 0 or (g <= 0 and bar_ha_new >= data['W']):
                    c = (data['theta1']*dst_o_d) - (data['theta'] * dst) + data['alpha_penalty'] * (rho_a + rho_b)
                    # parametric incentive to match drivers with low utilization
                else:
                    c = (data['theta1']*dst_o_d) - (data['theta'] * dst) + data['alpha_penalty'] * (rho_a + rho_b) \
                        - (penalty * rho_a)  # end of period penalty
# Note: right now only accounting for distance from driver to origin
                x_ab_cost[(a, b)] = c
                x_ab_time[(a, b)] = time
    # define decision variables: x_tab
    mdl.xab = mdl.continuous_var_dict(keys, lb=0)

    # define cost function
    mdl.match = mdl.sum(mdl.xab[(a, b)] * x_ab_cost[(a, b)] for a in drivers
                        for b in x_ab_feasible[('a', a)])
    mdl.maximize(mdl.match)

    # Add constraints in a separate function
    # resource constraint
    a_cnst = mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for b in x_ab_feasible[('a', a)]) == 1, 'Resource_a%d'
                                  % a) for a in drivers)

    # demand constraint
    mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for a in x_ab_feasible[('b', b)]) <= 1, 'Demand_b%d'
                         % b) for b in demand_list['active'])
    return mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst


# Setup model
def setup_model_base_case(data, t, demand_list, driver_list, driver_attr, demand_attr):
    model_name = 'LP_t%d' % t
    mdl = Model(name=model_name)
    # define decision variables: x_tab
    x_ab_feasible = {('b', b): [] for b in demand_list['active']}
    x_ab_cost, x_ab_time, x_ab_dist, keys = {}, {}, {}, []

    drivers = driver_list['act_available'] + driver_list['inactive']
    for a in drivers:
        x_ab_feasible[('a', a)] = [0]
        keys.append((a, 0))
        # obj func value when unmatched (matched to 0)
        x_ab_cost[(a, 0)] = 0

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

            if time <= demand_attr[b].delivery_deadline:  # if can be delivered before deadline
                keys.append((a, b))
                x_ab_feasible[('a', a)].append(b)
                x_ab_feasible[('b', b)].append(a)

                # perc = 1 - float(time / demand_attr[b].delivery_deadline)  # perc of time window remaining
                # rho_b = cal_rho_b(perc)

                c = (data['theta']*dst_o_d) - (data['theta'] * dst)  # + data['alpha_penalty'] * rho_b  # removed penalty of missed order
                # Note: right now only accounting for distance from driver to origin
                x_ab_cost[(a, b)] = c
                x_ab_time[(a, b)] = time

    # define decision variables: x_tab
    mdl.xab = mdl.continuous_var_dict(keys, lb=0)

    # define cost function
    mdl.match = mdl.sum(mdl.xab[(a, b)] * x_ab_cost[(a, b)] for a in drivers
                        for b in x_ab_feasible[('a', a)])
    mdl.maximize(mdl.match)

    # Add constraints in a separate function
    # resource constraint
    a_cnst = mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for b in x_ab_feasible[('a', a)]) == 1, 'Resource_a%d'
                                  % a) for a in drivers)

    # demand constraint
    mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for a in x_ab_feasible[('b', b)]) <= 1, 'Demand_b%d'
                         % b) for b in demand_list['active'])
    return mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst


# solve model
def solve_get_values(mdl, x_ab_feasible, a_cnst, driver_list, x_ab_dist):
    # solve model
    mdl.solve()
    # Get obj and var values
    z = mdl.objective_value

    print('z = ' + str(z))
    xab = {('b', 0): []}
    dual_a = {"all": mdl.dual_values(a_cnst)}
    dst, tot_dst = 0, 0

    # Get solution values
    num_matches = 0
    drivers = driver_list['act_available'] + driver_list['inactive']
    for i in range(len(drivers)):
        a = drivers[i]
        for b in x_ab_feasible[('a', a)]:
            if mdl.xab[(a, b)].solution_value >= 0.99:
                xab[('a', a)] = b
                if b != 0:
                    dst += x_ab_dist[(a, b)]
                    tot_dst += x_ab_dist[(a, b, 'tot_dst')]

                break  # break out of the b loop as soon as you find match
        if xab[('a', a)] != 0:
            xab[('b', b)] = a
            num_matches += 1
        else:
            xab[('b', 0)].append(a)  # driver unmatched
        # Get dual values
        dual_a[a] = dual_a["all"][i]
    print('num of matches = ' + str(num_matches))
    return xab, z, dual_a, num_matches, dst, tot_dst


# setup model then solve it
# Verified for 1. disagg alg, 2. agg alg, 3. Test_VFA_convergence_N
def solve_opt(data, t, demand_list, driver_list, driver_attr, demand_attr, base_case, cfa, penalty):
    if base_case:
        mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst = setup_model_base_case(data, t, demand_list, driver_list,
                                                            driver_attr, demand_attr)
    elif cfa:
        mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst = setup_model_cfa(data, t, demand_list, driver_list,
                                                                                 driver_attr, demand_attr, penalty)
    else:
        mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst = setup_model(data, t, demand_list, driver_list,
                                                        driver_attr, demand_attr, penalty)
    xab, z, dual_a, num_matches, dst, tot_dst = solve_get_values(mdl, x_ab_feasible, a_cnst, driver_list, x_ab_dist)
    return xab, z, dual_a, num_matches, x_ab_time, dst, tot_dst


# Step 3.4a: # Done
# This func updates driver and demand attributes after matching.
# Verified for 1. disagg alg, 2. agg alg, 3. Test_VFA_convergence_N
def update_attr(driver_list, demand_list, xab, driver_attr, demand_attr, t, data, x_ab_time, stats=False,
                driver_log=[], driver_stats=[], demand_stats=[]):
    # data = {'t_interval': [], 'h_int': [], 'Guaranteed_service': [], 'gamma': [], 'eta': [], 'W': [], 'T': []}
    driver_temp = {'act_av': [], 'act_unav': [], 'inact': []}
    drivers = driver_list['act_available'] + driver_list['inactive']
    for a in drivers:
        # if driver was not matched at t
        if xab[('a', a)] == 0:
            if driver_attr[a].sa == 0:  # if driver is inactive  # driver attr does not change
                driver_temp['inact'].append(a)
                if stats:
                    driver_log[a]['all_t'].append(['t%d_post-decision' % t, 'unmatched', 'inactive'])
            else:  # if driver was previously active
                if stats:
                    driver_log[a]['all_t'].append(['t%d_post-decision' % t, 'unmatched', 'active',
                                                   [driver_attr[a].ha, driver_attr[a].bar_ha]])
                if ((t+1)*data['t_interval'] - driver_attr[a].time_first_matched) >= data['Guaranteed_service']:
                    loc_d = driver_attr[a].loc  # Driver attribute resets, location does not change

                    if stats:
                        bar_ha = driver_attr[a].bar_ha
                        if bar_ha >= data['W']:
                            driver_stats['guarantee_met'].append([a, bar_ha])
                        else:
                            driver_stats['guarantee_missed'].append([a, bar_ha])

                    driver_attr[a] = Driver(a, 0, 1, loc_d, 0, 0, 0)
                    driver_temp['inact'].append(a)
                else:  # duration < guaranteed service (& driver was not matched)
                    busy_time = driver_attr[a].ha
                    if driver_attr[a].bar_ha == 0:  # if no prev busy time
                        tot_time = data['t_interval']
                        if busy_time > 1e-6:  # check
                            print('Possible error here! busy_time > tot_time')
                    else:
                        tot_time = (driver_attr[a].ha/driver_attr[a].bar_ha) + data['t_interval']
                    # total prev time(active time/perc active) + t interval
                    bar_ha = round(busy_time/tot_time*100/data['h_int'])*data['h_int']/100
                    if bar_ha > 1.0:
                        print('Stop and debug!')
                    driver_attr[a].bar_ha = bar_ha
                    driver_temp['act_av'].append(a)
                    driver_attr[a].iteration_history[t] = "This driver was not matched at t = " + str(t) \
                                                          + " and is in location " + str(driver_attr[a].loc) + "."
        else:  # Next: continue att update if driver was matched
            b = xab[('a', a)]
            if driver_attr[a].sa == 0:
                driver_attr[a].time_first_matched = t * data['t_interval']
                driver_attr[a].sa = 1

            # check if time it takes to make delivery will be before next t
            if x_ab_time[(a, b)] <= (t+1)*data['t_interval']:  # if fulfill delivery before next period
                driver_temp['act_av'].append(a)
            # if driver will still be delivering by t+1
            else:
                # unavailable, next time available is x_ab_time or possibly change entry 1 to x_ab_time right away
                driver_attr[a].ma = 0
                driver_attr[a].available_time = x_ab_time[(a, b)]
                driver_temp['act_unav'].append(a)

            busy_time = driver_attr[a].ha + (x_ab_time[(a, b)] - t * data['t_interval'])  # additional busy time t
            if driver_attr[a].bar_ha == 0:
                tot_time = (x_ab_time[(a, b)] - t * data['t_interval'])
            else:
                tot_time = (driver_attr[a].ha / driver_attr[a].bar_ha) + (x_ab_time[(a, b)] - t * data['t_interval'])
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
            if bar_ha > 1.0:
                print('Stop and debug!')
            driver_attr[a].bar_ha = bar_ha
            if stats:
                driver_log[a]['all_t'].append(['t%d_post_decision' % t,
                    [demand_attr[b].origin, demand_attr[b].destination, x_ab_time[(a, b)], busy_time,
                     driver_attr[a].bar_ha]])

    if t > 0:  # check if previously unavailable drivers are now available
        if len(driver_list['act_unavailable']) > 0:
            for a in driver_list['act_unavailable']:
                avail_time = driver_attr[a].available_time
                if avail_time <= (t+1)*data['t_interval']:  # if driver will be avail at t+1
                    if ((t+1)*data['t_interval'] - driver_attr[a].time_first_matched) >= data['Guaranteed_service']:
                        # driver attr resets if he meets the x hours threshold
                        loc = driver_attr[a].loc

                        if stats:
                            bar_ha = driver_attr[a].bar_ha
                            if bar_ha >= data['W']:
                                driver_stats['guarantee_met'].append([a, bar_ha])
                            else:
                                driver_stats['guarantee_missed'].append([a, bar_ha])
                        driver_attr[a] = Driver(a, 0, 1, loc, 0, 0, 0)
                        driver_temp['inact'].append(a)
                    else:  # done deliv, within guaranteed service  --> available, active
                        driver_attr[a].ma = 1
                        driver_temp['act_av'].append(a)
                else:  # if not done delivering, move it to unavailable t+1
                    driver_temp['act_unav'].append(a)
    if stats:
        driver_stats[t, 'active'] = len(driver_temp['act_av']) + len(driver_temp['act_unav'])

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
    return driver_temp, demand_temp


# Verified for 1. disagg alg, 2. agg alg, 3. Test_VFA_convergence_N
def predecision_state(num_drivers, num_demand, driver_temp, demand_temp, driver_list, demand_list, t, m_n, d_n,
                      driver_attr, demand_attr, data, stats=False, driver_log=[], driver_stats=[]):

    if stats and t + 1 >= data['T']:
        pass
    else:
        if m_n[t, 'exit'] > 0:
            num_inactive = len(driver_temp['inact'])
            samples = min(m_n[t, 'exit'], num_inactive)  # exit at end of t

            # decide on which drivers to exit randomly
            driver_exit = random.sample(driver_temp['inact'], samples)
            for a in driver_exit:
                driver_list['exit'].append(a)
                driver_temp['inact'].remove(a)
                if stats:
                    driver_log[a]['all_t'].append(['t%d' % t, 'exited'])

    def update_driver_attr(num_drivers):
        driver_list['act_available'] = copy.copy(driver_temp['act_av'])
        driver_list['act_unavailable'] = copy.copy(driver_temp['act_unav'])
        driver_list['inactive'] = copy.copy(driver_temp['inact'])

        # Let go of drivers who reached guarantee
        for a in driver_list['act_available']:
            if (t+1)*data["t_interval"] - driver_attr[a].time_first_matched >= data['Guaranteed_service']:
                loc = driver_attr[a].loc

                if stats:
                    bar_ha = driver_attr[a].bar_ha
                    if bar_ha >= data['W']:
                        driver_stats['guarantee_met'].append([a, bar_ha])
                    else:
                        driver_stats['guarantee_missed'].append([a, bar_ha])

                driver_attr[a] = Driver(a, 0, 1, loc, 0, 0, 0)
                driver_list['inactive'].append(a)
                driver_list['act_available'].remove(a)
                if stats:
                    driver_log[a]['all_t'].append(['t%d' % (t+1), [0, 1, loc, 0, 0, 0]])
            else:
                if stats:
                    driver_log[a]['all_t'].append(['t%d' % (t + 1), [driver_attr[a].sa, driver_attr[a].ma,
                     driver_attr[a].loc, driver_attr[a].time_first_matched, driver_attr[a].ha, driver_attr[a].bar_ha]])

        # new drivers
        if stats and t+1 >= data['T']:
            pass
        else:
            for j in range(m_n[t+1]):  # create new driver objects range(m[n][t+1])
                a = j+num_drivers
                loc = m_n[t+1, j]

                if stats:
                    driver_log[a] = {'all_t': ['t%d' % (t + 1), [0, 1, loc, 0, 0, 0]]}
                driver_attr[a] = Driver(a, 0, 1, loc, 0, 0, 0)

                driver_list['inactive'].append(a)
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
                demand_attr[b] = Demand(b, i_loc_o, i_loc_d, dst, announce_time, announce_time + data['delta_t'])
                demand_list['active'].append(b)
            num_demand += d_n[t+1]
        return num_demand

    num_drivers = update_driver_attr(num_drivers)
    num_demand = update_demand_attr(num_demand)
    return num_drivers, num_demand



