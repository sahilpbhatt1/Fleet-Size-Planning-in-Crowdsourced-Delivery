"""
MCDRP Batch Experiment Runner

This script orchestrates batch experiments for the Multi-Commodity Delivery
Routing Problem solver. It runs multiple scenarios with different configurations:

- Network sizes (N = 30, 50, etc.)
- Beta parameters (supply-demand balance)
- Model types (UB: upper bound, LB: lower bound)
- Demand disturbance settings

Experimental Setup:
    - Uses Chicago synthetic network data
    - Tests different supply scenarios
    - Outputs results for analysis and comparison

Usage:
    python run_all_Sahil.py

Author: Sahil Bhatt
"""

import Model_unified_same_zone_matching_Sahil as unified_same_zone

# =============================================================================
# EXPERIMENTAL CONFIGURATION
# =============================================================================

# Test scenarios
supply_type_list = [0]
inst_list = range(1)  # Instance numbers to run
cnt_m = 0

# Supply-demand balance parameter (beta)
# 0.5 = undersupply, 1 = balanced, 2 = oversupply
beta = 1
enter_exit = False  # Whether to model driver entry/exit dynamics

# Network configurations to test
network_sizes = [30]  # N = number of zones

for N in network_sizes:
    # Set beta string for output file naming
    if beta == 0.5:
        beta_string = '_half_beta'
    elif beta == 1:
        beta_string = '_one_beta'
    elif beta == 2:
        beta_string = '_double_beta'
    else:
        beta_string = f'_beta{beta}'
    
    # Load network data
    data_name = f'data/chicago-synthetic-inst-N{N}_v4.npz'
    
    # Model configurations
    model_type_list = ['UB']  # Upper bound model
    disturb_list = [False]  # No demand disturbance
    ee_str = ''  # Enter/exit string for naming
    
    print("=" * 60)
    print(f"RUNNING MCDRP EXPERIMENTS: N={N}, beta={beta}")
    print("=" * 60)
    
    for model_type in model_type_list:
        # Skip invalid configurations
        if enter_exit and model_type == 'LB':
            continue
            
        for disturb in disturb_list:
            dist_str = '_disturb' if disturb else ''
            inst_string = f'{ee_str}_{model_type}{beta_string}{dist_str}'
            
            print(f"\nConfiguration: {inst_string}")
            print(f"  - Model type: {model_type}")
            print(f"  - Disturbance: {disturb}")
            print(f"  - Data file: {data_name}")
            
            # Run the solver
            unified_same_zone.solve_sequentially(
                inst_string,
                inst_list,
                beta,
                model_type,
                True,  # verbose
                data_name,
                enter_exit,
                disturb,
                cnt_m
            )
            
    print(f"\n✓ Completed experiments for N={N}")
