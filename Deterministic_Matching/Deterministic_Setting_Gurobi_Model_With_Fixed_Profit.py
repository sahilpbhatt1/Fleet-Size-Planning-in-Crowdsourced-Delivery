"""
Driver-Order Matching: Single-Period Deterministic Model

This module implements a SIMPLIFIED version of the matching subproblem from:
    "Fleet Size Planning in Crowdsourced Delivery: 
     Balancing Service Level and Driver Utilization"

PURPOSE:
    This code solves the INNER LOOP of the fleet size planning problem:
    Given a fixed set of drivers and orders, find the optimal assignment.
    
    The full problem (in Deterministic-Matching-Policy/) also decides:
    - How many drivers to activate (pool sizing)
    - How to handle stochastic demand arrivals
    - How to enforce wage guarantee constraints
    
    This module isolates just the matching step for clarity and testing.

PROBLEM FORMULATION (Bipartite Matching):

    INPUTS:
        - m drivers at locations {d_1, d_2, ..., d_m}
        - n orders, each with (origin, destination, revenue)
    
    DECISION VARIABLES:
        - x[i,j] ∈ {0,1}: 1 if driver i assigned to order j
    
    OBJECTIVE:
        maximize Σ (revenue[j] - cost[i,j]) * x[i,j]
        
        where cost[i,j] = distance(driver_i, origin_j) + distance(origin_j, dest_j)
    
    CONSTRAINTS:
        - Each driver assigned to at most one order:
          Σ_j x[i,j] ≤ 1  for all drivers i
          
        - Each order fulfilled by at most one driver:
          Σ_i x[i,j] ≤ 1  for all orders j

WHY "DETERMINISTIC"?
    - All inputs (driver locations, orders) are KNOWN at decision time
    - No uncertainty about future demand
    - Contrast with stochastic models that consider demand scenarios

WHY "FIXED PROFIT"?
    - Order revenue is fixed (not dependent on which driver serves it)
    - Variable_Driver_Reliability.py extends this with driver-dependent profits

VISUALIZATION:
    The code includes NetworkX graph visualization showing:
    - Drivers as blue nodes
    - Orders as red nodes  
    - Optimal assignments as green edges

This is useful for:
    - Understanding the matching subproblem in isolation
    - Testing/debugging the MIP formulation
    - Demonstrating optimization to non-experts

Dependencies:
    - gurobipy: Gurobi optimization solver
    - numpy: Numerical computations
    - scipy: Distance calculations
    - networkx, matplotlib: Graph visualization

Author: Sahil Bhatt
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt 
import networkx as nx
from typing import List, Tuple, Optional


def generate_inst(num_drivers: int, num_orders: int) -> Tuple[List, List, List]:
    """
    Generate a random problem instance for driver-order matching.
    
    Creates synthetic test data by placing drivers and orders randomly
    on a 10x10 grid (representing a city).
    
    Args:
        num_drivers: Number of available drivers
        num_orders: Number of delivery orders
        
    Returns:
        Tuple of:
            - driver_locations: [[x,y], ...] for each driver
            - order_locations: [None, [[origin], [dest]], ...] 
              (index 0 = None represents "unassigned" state)
            - order_revenues: [None, rev1, rev2, ...] revenue per order
            
    Note:
        Index 0 is reserved for the "unassigned" dummy order,
        allowing the model to leave drivers idle if profitable.
    """
    # Define discrete location set on 10x10 grid (100 possible locations)
    loc_set = [[0.5 + x, 0.5 + y] for x in range(10) for y in range(10)]
    
    def closest_node(node: List[float]) -> List[float]:
        """Map continuous coordinates to nearest discrete grid point."""
        nodes = np.asarray(loc_set)
        closest_ind = distance.cdist([node], nodes).argmin() 
        return loc_set[closest_ind]

    # Generate random driver locations
    driver_locations = [
        closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)]) 
        for _ in range(num_drivers)
    ]
 
    # Initialize order data (index 0 = None represents unassigned state)
    order_locations = [None]
    order_revenues = [None]
    
    # Revenue bounds for order profitability
    min_revenue, max_revenue = 11, 21
    
    # Generate random orders with origin-destination pairs
    for _ in range(num_orders):
        origin = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        destination = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        
        # Ensure origin and destination are distinct
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
    Calculate total cost for driver i to fulfill order j.
    
    Cost = (distance to pickup) + (distance to deliver)
    
    This represents:
        - Fuel/vehicle costs
        - Driver time (opportunity cost)
        - Platform pays per-mile for delivery
    """
    driver_loc = driver_locations[driver_idx]
    order_origin = order_locations[order_idx][0]
    order_dest = order_locations[order_idx][1]
    
    # Pickup cost: driver to order origin
    pickup_distance = euclidean_distance(driver_loc, order_origin)
    
    # Delivery cost: order origin to destination
    delivery_distance = euclidean_distance(order_origin, order_dest)
    
    return pickup_distance + delivery_distance

if __name__ == '__main__':
    """
    Main execution: Solve driver-order matching optimization and visualize results.
    
    This script demonstrates:
        1. Problem instance generation with random drivers and orders
        2. MIP model formulation using Gurobi
        3. Optimal solution computation
        4. Bipartite graph visualization of the matching
    """
    
    # Problem parameters
    num_drivers = 10
    num_orders = 10
    
    # Generate random instance
    driver_locations, order_locations, order_revenues = generate_inst(num_drivers, num_orders)
    num_orders += 1  # Account for None placeholder at index 0
    
    print("=" * 60)
    print("DRIVER-ORDER MATCHING OPTIMIZATION")
    print("=" * 60)
    print(f"\nDriver Locations: {driver_locations}")
    print(f"\nOrder Locations: {order_locations}")
    print(f"\nOrder Revenues: {order_revenues}")
    
    # =========================================================================
    # GUROBI OPTIMIZATION MODEL
    # =========================================================================
    
    model = gp.Model('DriverOrderMatching')
    
    # Decision Variables: x[i,j] = 1 if driver i assigned to order j
    # Note: j=0 represents "unassigned" state
    assign = model.addVars(
        [(i, j) for i in range(num_drivers) for j in range(num_orders)],
        name='assign',
        vtype=GRB.BINARY
    )
    
    # Constraint 1: Each driver must be assigned exactly once
    # (either to an order or to the "unassigned" state j=0)
    model.addConstrs(
        (gp.quicksum(assign[i, j] for j in range(num_orders)) == 1 
         for i in range(num_drivers)),
        name='driver_assignment'
    )
    
    # Constraint 2: Each order can be fulfilled by at most one driver
    # (excludes j=0 which is the unassigned state)
    model.addConstrs(
        (gp.quicksum(assign[i, j] for i in range(num_drivers)) <= 1 
         for j in range(1, num_orders)),
        name='order_capacity'
    )
    
    # Objective: Maximize total profit (revenue - delivery cost)
    model.setObjective(
        gp.quicksum(
            (order_revenues[j] - calculate_delivery_cost(i, j, driver_locations, order_locations)) * assign[i, j]
            for i in range(num_drivers) 
            for j in range(1, num_orders)
        ),
        GRB.MAXIMIZE
    )
    
    # Solve the optimization model
    print("\n" + "=" * 60)
    print("SOLVING OPTIMIZATION MODEL...")
    print("=" * 60)
    model.optimize()
    
    # =========================================================================
    # VISUALIZATION: Bipartite Matching Graph
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("GENERATING MATCHING VISUALIZATION...")
    print("=" * 60)
    
    # Create bipartite graph
    G = nx.Graph()
    
    # Add driver and order nodes
    driver_nodes = [f'Driver {i}' for i in range(len(driver_locations))][::-1]
    order_nodes = [f'Order {i}' for i in range(len(order_locations))][::-1]
    
    G.add_nodes_from(driver_nodes, bipartite=0)
    G.add_nodes_from(order_nodes, bipartite=1)
    
    # Add edges for optimal matches
    matched_edges = [
        (f'Driver {i}', f'Order {j}') 
        for i in range(num_drivers) 
        for j in range(num_orders) 
        if assign[i, j].x == 1
    ]
    G.add_edges_from(matched_edges)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.title('Optimal Driver-Order Matching', fontsize=14, fontweight='bold')
    
    # Position nodes in two columns
    pos = {}
    pos.update((node, (1, i * 2)) for i, node in enumerate(driver_nodes))
    pos.update((node, (3, i * 2)) for i, node in enumerate(order_nodes))
    
    # Draw the graph
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
    
    print("\nOptimization complete. Matching visualization displayed.")
