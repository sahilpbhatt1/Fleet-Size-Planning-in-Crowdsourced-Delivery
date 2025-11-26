
# This code generates random instances for dynamic matching
# Go to below link for report
# C:\Users\aliaa\OneDrive - University of Waterloo\Research\Crowdsourced delivery project 2\Latex - Oct

import numpy as np
import itertools
import math
from scipy.spatial import distance


def generate_inst():
    loc_set = [[0.5 + x, 0.5 + y] for x in range(10) for y in range(10)]

    def closest_node(node):
        nodes = np.asarray(loc_set)
        closest_ind = distance.cdist([node], nodes).argmin()
        return loc_set[closest_ind]

    # iter 1, t=0
    m = {}        # This dict stores the no. of drivers and their location/dict for first iter
    d = {}        # Dict stores no. of orders and their o-d locations
    m['exit'] = {}

    for t in range(T):
        # Driver info
        m_t = np.random.poisson(mu_enter)  # realization of random driver enterance
        m['exit'][t] = np.random.poisson(mu_exit)
        m[t] = m_t

        if m_t > 0:
            for i in range(m_t):
                # set random location of drivers on 10x10 grid
                m[t, i] = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])

        # Demand info
        d_t = np.random.poisson(lambd)
        d[t] = d_t
        if d_t > 0:
            for i in range(d_t):
                # set random origin and destination locations of drivers on 10x10 grid,
                # random announcement time between 2 epochs
                o_discrete = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])
                d_discrete = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])
                while o_discrete == d_discrete:  # to make sure origin != dest
                    d_discrete = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])
                d[t, i] = [[o_discrete, d_discrete], np.random.uniform((t-1)*10, t*10)]
    return m, d


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
    alpha = 1
    for inst in range(1, 11):
        m, d = generate_inst()

        name = 'data/test_inst_T=%d_%d_mu%d' % (T, inst, mu_enter)
        np.savez(name, T=T, m=m, d=d, mu_enter=mu_enter, mu_exit=mu_exit, lambd=lambd, delta_t=delta_t,
                 t_interval=t_interval, theta=theta, min_guarantee=min_guarantee, alpha=alpha, rho_a=rho_a, rho_b=rho_b)
