"""
Stochastic Demand and Supply Generation for Crowdsourced Delivery

This module generates synthetic instances for dynamic driver-order matching
problems in crowdsourced delivery platforms. It simulates:

1. Driver arrivals/departures following Poisson processes
2. Order (demand) arrivals with random origin-destination pairs  
3. Time-varying demand patterns across planning horizons

The generated data is used for:
- Training and testing optimization algorithms
- Validating reinforcement learning policies
- Benchmarking workforce scheduling approaches

Stochastic Process Parameters:
    - mu_enter: Mean driver arrival rate (Poisson)
    - mu_exit: Mean driver departure rate (Poisson)
    - lambda: Mean demand arrival rate (Poisson)
    - T: Total number of time epochs (planning horizon)

Author: Sahil Bhatt
"""

import numpy as np
import itertools
import math
from scipy.spatial import distance
from typing import Dict, Tuple, List


def generate_instance(
    T: int,
    mu_enter: float,
    mu_exit: float,
    lambd: float,
    t_interval: float,
    grid_size: int = 10
) -> Tuple[Dict, Dict]:
    """
    Generate a stochastic instance of driver-order data over T time epochs.
    
    Simulates a Poisson process for both driver arrivals/departures and
    demand (order) arrivals. Locations are discretized on a grid.
    
    Args:
        T: Number of time epochs in planning horizon
        mu_enter: Mean arrival rate for new drivers (Poisson parameter)
        mu_exit: Mean departure rate for drivers leaving (Poisson parameter)
        lambd: Mean demand arrival rate per epoch (Poisson parameter)
        t_interval: Duration of each time epoch (minutes)
        grid_size: Size of the discrete location grid (grid_size x grid_size)
        
    Returns:
        Tuple containing:
            - m: Dictionary with driver information
                - m[t]: Number of drivers entering at epoch t
                - m[t, i]: Location of driver i entering at epoch t
                - m['exit'][t]: Number of drivers exiting at epoch t
            - d: Dictionary with demand information
                - d[t]: Number of orders arriving at epoch t
                - d[t, i]: Order details [[origin, destination], announce_time]
                
    Example:
        >>> m, d = generate_instance(T=192, mu_enter=2, mu_exit=4, lambd=10, t_interval=5)
        >>> print(f"Drivers entering at t=0: {m[0]}")
        >>> print(f"Orders at t=0: {d[0]}")
    """
    # Create discrete location set on grid
    loc_set = [[0.5 + x, 0.5 + y] for x in range(grid_size) for y in range(grid_size)]

    def closest_node(node: List[float]) -> List[float]:
        """Map continuous coordinates to nearest discrete grid point."""
        nodes = np.asarray(loc_set)
        closest_ind = distance.cdist([node], nodes).argmin()
        return loc_set[closest_ind]

    # Initialize data structures
    m = {}  # Driver data
    d = {}  # Demand data
    m['exit'] = {}

    for t in range(T):
        # =====================================================================
        # DRIVER SUPPLY GENERATION (Poisson Process)
        # =====================================================================
        m_t = np.random.poisson(mu_enter)  # Drivers entering at epoch t
        m['exit'][t] = np.random.poisson(mu_exit)  # Drivers exiting at epoch t
        m[t] = m_t

        if m_t > 0:
            for i in range(m_t):
                # Assign random location on grid for each entering driver
                m[t, i] = closest_node([
                    np.random.uniform(0, grid_size), 
                    np.random.uniform(0, grid_size)
                ])

        # =====================================================================
        # DEMAND GENERATION (Poisson Process)
        # =====================================================================
        d_t = np.random.poisson(lambd)
        d[t] = d_t
        
        if d_t > 0:
            for i in range(d_t):
                # Generate random origin-destination pair
                origin = closest_node([
                    np.random.uniform(0, grid_size), 
                    np.random.uniform(0, grid_size)
                ])
                destination = closest_node([
                    np.random.uniform(0, grid_size), 
                    np.random.uniform(0, grid_size)
                ])
                
                # Ensure origin != destination
                while origin == destination:
                    destination = closest_node([
                        np.random.uniform(0, grid_size), 
                        np.random.uniform(0, grid_size)
                    ])
                
                # Random announcement time within epoch interval
                announce_time = np.random.uniform((t - 1) * t_interval, t * t_interval)
                d[t, i] = [[origin, destination], announce_time]
                
    return m, d


if __name__ == '__main__':
    """
    Generate and save multiple test instances for experimental evaluation.
    """
    
    # ==========================================================================
    # SIMULATION PARAMETERS
    # ==========================================================================
    T = 192                # Planning horizon: 16 hours (192 epochs of 5 min each)
    mu_enter = 2           # Mean driver arrival rate
    mu_exit = 4            # Mean driver departure rate
    lambd = 10             # Mean demand arrival rate
    t_interval = 5         # Minutes per epoch
    delta_t = t_interval * 18  # Delivery time window (90 minutes)
    
    # Model parameters
    theta = 1 / (2 * math.sqrt(200))  # Distance cost coefficient
    rho_a = [[0, 0.8, 1], [1, 0, 0]]  # Driver priority score breakpoints
    rho_b = [[[0, 0.24999999], [0.5, 0.5]], [[0.25, 0.5, 1], [0.25, 0, 0]]]  # Demand priority
    min_guarantee = 0.8    # Minimum utilization guarantee for drivers
    alpha = 1              # Objective function weight
    
    print("=" * 60)
    print("GENERATING TEST INSTANCES FOR CROWDSOURCED DELIVERY")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  - Planning horizon: {T} epochs ({T * t_interval / 60:.1f} hours)")
    print(f"  - Driver arrival rate: μ_enter = {mu_enter}")
    print(f"  - Driver departure rate: μ_exit = {mu_exit}")
    print(f"  - Demand arrival rate: λ = {lambd}")
    
    # Generate multiple instances for robust experimental evaluation
    num_instances = 10
    
    for inst in range(1, num_instances + 1):
        m, d = generate_instance(T, mu_enter, mu_exit, lambd, t_interval)
        
        # Compute instance statistics
        total_drivers = sum(m[t] for t in range(T))
        total_orders = sum(d[t] for t in range(T))
        
        filename = f'data/test_inst_T={T}_{inst}_mu{mu_enter}'
        
        np.savez(
            filename,
            T=T, m=m, d=d,
            mu_enter=mu_enter, mu_exit=mu_exit, lambd=lambd,
            delta_t=delta_t, t_interval=t_interval,
            theta=theta, min_guarantee=min_guarantee, alpha=alpha,
            rho_a=rho_a, rho_b=rho_b
        )
        
        print(f"  Instance {inst}: {total_drivers} drivers, {total_orders} orders → {filename}.npz")
    
    print(f"\n✓ Successfully generated {num_instances} test instances")
