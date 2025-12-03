"""
Driver-Order Matching with Variable Driver Reliability

This module extends the basic driver-order matching model to incorporate
driver reliability factors. The reliability score affects the expected
revenue from each match, accounting for real-world variability in gig
worker performance.

Mathematical Formulation:
    maximize    Σ(revenue[j] * reliability[i] - cost[i,j]) * x[i,j]
    subject to  Σ_j x[i,j] = 1           ∀ drivers i
                Σ_i x[i,j] ≤ 1           ∀ orders j
                x[i,j] ∈ [0,1]           Continuous relaxation for LP

Key Extensions from Base Model:
    - Driver reliability scores modulate expected revenue
    - Linear Programming relaxation (continuous variables)
    - Analysis of reliability impact on optimal matching

Applications:
    - Workforce scheduling with heterogeneous worker quality
    - Resource allocation under performance uncertainty
    - Service quality optimization in gig economy platforms

Author: Sahil Bhatt
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt 
import networkx as nx
from typing import List, Tuple


def generate_inst(num_drivers: int, num_orders: int) -> Tuple[List, List, List]:
    """
    Generate a random problem instance with driver and order locations.
    
    Args:
        num_drivers: Number of available drivers
        num_orders: Number of delivery orders
        
    Returns:
        Tuple of (driver_locations, order_locations, order_revenues)
    """
    loc_set = [[0.5 + x, 0.5 + y] for x in range(10) for y in range(10)]
    
    def closest_node(node: List[float]) -> List[float]:
        """Map continuous coordinates to nearest discrete grid point."""
        nodes = np.asarray(loc_set)
        closest_ind = distance.cdist([node], nodes).argmin() 
        return loc_set[closest_ind]

    driver_locations = [
        closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)]) 
        for _ in range(num_drivers)
    ]
    
    order_locations = [None]  # Index 0 = unassigned state
    order_revenues = [None]
    min_revenue, max_revenue = 11, 21
    
    for _ in range(num_orders):
        origin = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        destination = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        
        while origin == destination:
            destination = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])

        order_locations.append([origin, destination])
        order_revenues.append(np.random.randint(min_revenue, max_revenue + 1))
        
    return driver_locations, order_locations, order_revenues


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate Euclidean distance between two points."""
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


def calculate_delivery_cost(driver_idx: int, order_idx: int,
                           driver_locations: List, order_locations: List) -> float:
    """
    Calculate total delivery cost for assigning a driver to an order.
    
    Cost = pickup_distance + delivery_distance
    """
    driver_loc = driver_locations[driver_idx]
    order_origin = order_locations[order_idx][0]
    order_dest = order_locations[order_idx][1]
    
    pickup_distance = euclidean_distance(driver_loc, order_origin)
    delivery_distance = euclidean_distance(order_origin, order_dest)
    
    return pickup_distance + delivery_distance
  
if __name__ == '__main__':
    """
    Main execution: Solve driver-order matching with variable reliability scores.
    
    This demonstrates the impact of driver reliability on optimal matching decisions.
    Uses LP relaxation (continuous variables) to analyze fractional solutions.
    """
    
    # Problem parameters
    num_drivers = 10
    num_orders = 20
    
    # Generate instance (or use pre-defined test case)
    driver_locations, order_locations, order_revenues = generate_inst(num_drivers, num_orders)
    num_orders += 1  # Account for None placeholder at index 0
    
    # Pre-defined test instance for reproducibility
    driver_locations = [
        [4.5, 0.5], [8.5, 9.5], [1.5, 0.5], [8.5, 7.5], [3.5, 5.5],
        [7.5, 0.5], [8.5, 6.5], [9.5, 4.5], [1.5, 0.5], [1.5, 6.5]
    ]
    order_locations = [
        None, [[1.5, 1.5], [5.5, 8.5]], [[5.5, 2.5], [3.5, 9.5]], 
        [[9.5, 5.5], [9.5, 3.5]], [[1.5, 4.5], [9.5, 1.5]], 
        [[4.5, 8.5], [9.5, 3.5]], [[9.5, 8.5], [1.5, 1.5]], 
        [[4.5, 2.5], [2.5, 8.5]], [[0.5, 9.5], [6.5, 7.5]], 
        [[4.5, 8.5], [7.5, 6.5]], [[8.5, 9.5], [3.5, 9.5]], 
        [[7.5, 8.5], [3.5, 4.5]], [[5.5, 1.5], [0.5, 1.5]], 
        [[0.5, 9.5], [6.5, 3.5]], [[1.5, 8.5], [6.5, 2.5]], 
        [[2.5, 0.5], [3.5, 6.5]], [[0.5, 0.5], [0.5, 3.5]], 
        [[2.5, 4.5], [4.5, 3.5]], [[7.5, 2.5], [4.5, 1.5]], 
        [[7.5, 7.5], [8.5, 8.5]], [[3.5, 2.5], [2.5, 4.5]]
    ]
    order_revenues = [
        None, 21, 11, 19, 20, 21, 20, 16, 18, 21, 14, 
        14, 18, 13, 13, 13, 19, 18, 19, 20, 14
    ]
    
    # Generate driver reliability scores (heterogeneous worker quality)
    # Reliability ∈ [0.7, 1.5] models high-performing to inconsistent workers
    driver_reliability = [np.random.uniform(0.7, 1.5) for _ in range(len(driver_locations))]
    
    print("=" * 60)
    print("DRIVER-ORDER MATCHING WITH VARIABLE RELIABILITY")
    print("=" * 60)
    print(f"\nDriver Locations: {driver_locations}")
    print(f"\nOrder Revenues: {order_revenues}")
    print(f"\nDriver Reliability Scores: {[round(r, 3) for r in driver_reliability]}")
    
    # =========================================================================
    # GUROBI LP MODEL (Continuous Relaxation)
    # =========================================================================
    
    model = gp.Model('DriverOrderMatchingWithReliability')
    
    # Decision Variables: Continuous for LP relaxation (analyze fractional solutions)
    assign = model.addVars(
        [(i, j) for i in range(num_drivers) for j in range(num_orders)],
        name='assign',
        vtype=GRB.CONTINUOUS,
        ub=1
    )
    
    # Constraint 1: Each driver assigned exactly once
    model.addConstrs(
        (gp.quicksum(assign[i, j] for j in range(num_orders)) == 1 
         for i in range(num_drivers)),
        name='driver_assignment'
    )
    
    # Constraint 2: Each order fulfilled by at most one driver
    model.addConstrs(
        (gp.quicksum(assign[i, j] for i in range(num_drivers)) <= 1 
         for j in range(1, num_orders)),
        name='order_capacity'
    )
    
    # Objective: Maximize reliability-weighted profit
    # Expected revenue = base_revenue * reliability_score
    model.setObjective(
        gp.quicksum(
            (order_revenues[j] * driver_reliability[i] - 
             calculate_delivery_cost(i, j, driver_locations, order_locations)) * assign[i, j]
            for i in range(num_drivers) 
            for j in range(1, num_orders)
        ),
        GRB.MAXIMIZE
    )
    
    # Solve
    print("\n" + "=" * 60)
    print("SOLVING LP RELAXATION...")
    print("=" * 60)
    model.optimize()
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    G = nx.Graph()
    
    driver_nodes = [f'Driver {i}' for i in range(len(driver_locations))][::-1]
    order_nodes = [f'Order {i}' for i in range(len(order_locations))][::-1]
    
    G.add_nodes_from(driver_nodes, bipartite=0)
    G.add_nodes_from(order_nodes, bipartite=1)
    
    # Add edges for matches (threshold at 0.5 for fractional solutions)
    matched_edges = [
        (f'Driver {i}', f'Order {j}') 
        for i in range(num_drivers) 
        for j in range(num_orders) 
        if assign[i, j].x >= 0.5
    ]
    G.add_edges_from(matched_edges)
    
    plt.figure(figsize=(12, 10))
    plt.title('Optimal Matching with Reliability-Weighted Objective', fontsize=14, fontweight='bold')
    
    pos = {}
    pos.update((node, (1, i * 2)) for i, node in enumerate(driver_nodes))
    pos.update((node, (3, i * 2)) for i, node in enumerate(order_nodes))
    
    nx.draw_networkx(
        G, pos=pos,
        node_size=300,
        node_color=['lightblue' if 'Driver' in n else 'lightgreen' for n in G.nodes()],
        edge_color='red',
        width=2,
        font_size=8
    )
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
