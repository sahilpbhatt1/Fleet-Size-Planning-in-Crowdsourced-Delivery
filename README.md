# Fleet Size Planning in Crowdsourced Delivery: Balancing Service Level and Driver Utilization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Gurobi](https://img.shields.io/badge/Gurobi-Optimization-red.svg)](https://www.gurobi.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Research Implementation**: Code for the paper *"Fleet Size Planning in Crowdsourced Delivery: Balancing Service Level and Driver Utilization"*

---

## The Research Problem

Crowdsourced delivery platforms (e.g., DoorDash, Instacart, Uber Eats) face a fundamental **trade-off**: they need to maintain enough drivers to fulfill customer orders quickly (high **service level**), but having too many drivers means each driver gets fewer deliveries and lower earnings (poor **driver utilization**).

This project develops optimization algorithms to solve the **fleet size planning problem**:

> **How many drivers should the platform activate in each planning period to maximize demand fulfillment while ensuring drivers earn enough to meet minimum wage guarantees?**

### Why This Matters

| Stakeholder | Concern | Our Solution |
|------------|---------|--------------|
| **Platform** | Fulfill as many orders as possible | Optimize pool sizes to match predicted demand |
| **Customers** | Fast, reliable deliveries | Maintain adequate driver supply for service level |
| **Drivers** | Earn minimum wage guarantee | Utilization-aware matching ensures drivers meet thresholds |

---

## Key Concepts

### Pool Size Decision
At the start of each planning period (e.g., hourly), the platform decides how many drivers to "activate" from a pool of available gig workers. Too few → unmet demand. Too many → underutilized drivers.

### Wage Guarantee Constraint
Many platforms offer drivers a minimum earnings guarantee (e.g., $15/hour). A driver "meets the guarantee" if their **utilization rate** (time spent delivering / total active time) exceeds a threshold W (typically 80%).

### The Trade-off
- **Larger pool size** → Higher demand fulfillment, but lower driver utilization
- **Smaller pool size** → Higher driver utilization, but more unfulfilled orders

Our algorithms find the **optimal balance** through mathematical optimization.

---

## Project Structure: Three Approaches

This repository contains **three distinct implementations**, each solving the fleet size planning problem with different techniques:

```
Fleet-Size-Planning-in-Crowdsourced-Delivery/
│
├── 📁 Deterministic_Matching/          [APPROACH 1: Single-Period Optimization]
│   │
│   │   Core driver-order matching models that solve the assignment problem
│   │   for a SINGLE time period with KNOWN demand (deterministic setting).
│   │
│   ├── Deterministic_Setting_Gurobi_Model_With_Fixed_Profit.py
│   │       → Basic bipartite matching: assigns drivers to orders to maximize profit
│   │       → Demonstrates MIP formulation with Gurobi solver
│   │
│   ├── Deterministic_Setting_Gurobi_Model_With_Variable_Driver_Reliability.py
│   │       → Extended model accounting for driver reliability (no-show probability)
│   │
│   ├── Generate_Random_data.py
│   │       → Synthetic data generator for testing (driver/order locations, revenues)
│   │
│   ├── Myopic_functions.py & Test_Myopic_collect_stats.py
│   │       → Myopic (greedy) policy baseline for comparison
│   │
│   ├── Multiple decision epochs/
│   │       → Extension to multi-period setting (still deterministic demand)
│   │
│   └── Reinforcement_Learning/
│       └── Temporal difference learning to solve TSP.ipynb
│               → Q-Learning demonstration on TSP (educational, shows RL concepts)
│
├── 📁 Deterministic-Matching-Policy/   [APPROACH 2: Multi-Period with Wage Guarantees]
│   │
│   │   MAIN IMPLEMENTATION of the paper's algorithm. Solves the fleet size
│   │   planning problem over MULTIPLE periods with STOCHASTIC demand and
│   │   DRIVER WAGE GUARANTEE constraints.
│   │
│   ├── Functions.py
│   │       → Core algorithms:
│   │         • boltzmann(): Softmax policy for pool size selection
│   │         • sample_path(): Poisson demand/supply simulation
│   │         • solve_opt(): Real-time LP matching with utilization tracking
│   │         • Driver/Demand classes: State tracking for guarantee enforcement
│   │         • cal_rho_a(): Priority scoring for underutilized drivers
│   │
│   ├── Main.py
│   │       → Training/evaluation loop:
│   │         • Iterates through N simulation runs
│   │         • Tracks: demand fulfillment rate, fraction meeting wage guarantee
│   │         • Outputs confidence intervals for statistical significance
│   │
│   ├── Generate_Random_data_wage_guarantee.py
│   │       → Instance generator with wage guarantee parameters
│   │
│   └── data/
│           → Pre-generated test instances (T=192 epochs = 16 hours)
│
├── 📁 Stochastic-Zone-Based-Routing/   [APPROACH 3: Spatial Decomposition]
│   │   (Renamed from "MCDRP code" for clarity)
│   │
│   │   Advanced model that adds GEOGRAPHIC ZONES to the problem. Drivers
│   │   can only serve orders within their zone or adjacent zones. Useful
│   │   for large cities with distinct neighborhoods.
│   │
│   ├── Model_unified_same_zone_matching_Sahil.py
│   │       → Multi-stage stochastic program with:
│   │         • Zone-based matching constraints (drivers serve nearby areas)
│   │         • Scenario decomposition for demand uncertainty
│   │         • Driver repositioning between zones
│   │
│   └── run_all_Sahil.py
│           → Batch experiment runner for parameter sweeps
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How the Approaches Differ

| Aspect | Deterministic Matching | Deterministic-Matching-Policy | Stochastic-Zone-Based-Routing |
|--------|----------------------|------------------------------|------------------------------|
| **Time Horizon** | Single period | Multi-period (16 hours) | Multi-period |
| **Demand Model** | Known (deterministic) | Stochastic (Poisson) | Stochastic (scenarios) |
| **Geography** | Flat (no zones) | Flat (no zones) | Zone-based network |
| **Wage Guarantee** | Not considered | ✅ Core feature | Not primary focus |
| **Algorithm** | MIP (Gurobi) | Softmax Policy + LP | Stochastic MIP |
| **Use Case** | Baseline/benchmark | **Main algorithm** | Large-scale cities |

---

## The Algorithm (Deterministic-Matching-Policy)

### High-Level Flow

```
For each planning period p = 1, 2, ..., 16:
    1. DECIDE pool size x[p] using Softmax policy
    2. SIMULATE driver arrivals (binomial from pool)
    3. For each epoch t in period p:
        a. OBSERVE new demand arrivals (Poisson)
        b. SOLVE matching LP: assign drivers to orders
           - Maximize: platform profit + utilization incentives
           - Subject to: driver capacity, demand coverage, time windows
        c. UPDATE driver states (location, utilization, earnings)
    4. TRACK metrics: demand fulfilled, drivers meeting guarantee
    5. UPDATE value function V[p, x] based on outcomes
```

### Key Innovation: Utilization-Aware Matching

The matching objective includes a **priority term** ρ_a that gives underutilized drivers higher priority:

```python
# Priority function: higher score for drivers below 80% utilization
def cal_rho_a(bar_ha):  # bar_ha = current utilization rate
    if bar_ha < 0.8:    # Below wage guarantee threshold
        return -1.25 * bar_ha + 1  # Higher priority
    return 0  # Already meeting guarantee
```

This ensures the platform **preferentially assigns orders to drivers who need more work** to meet their wage guarantee.

### Softmax Policy for Pool Sizing

Instead of using a fixed pool size, the algorithm **learns optimal pool sizes** through a value function V[period, pool_size]:

```python
# Boltzmann/Softmax selection with temperature annealing
P(pool_size = k) ∝ exp(-ζ · V[period, k])
```

- Early training: High temperature → exploration (try different pool sizes)
- Later training: Low temperature → exploitation (use best-performing sizes)

---

## Performance Metrics

The algorithm optimizes for two competing objectives:

1. **Demand Fulfillment Rate** = (Orders Fulfilled) / (Total Orders)
2. **Wage Guarantee Rate** = (Drivers Meeting Threshold) / (Total Drivers)

Results show the trade-off frontier—as we increase pool size, fulfillment rises but guarantee rate falls.

---

## Installation & Usage

### Prerequisites
- Python 3.8+
- Gurobi Optimizer (free academic license at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/))

### Setup

```bash
git clone https://github.com/sahilpbhatt1/Fleet-Size-Planning-in-Crowdsourced-Delivery.git
cd Fleet-Size-Planning-in-Crowdsourced-Delivery
pip install -r requirements.txt
```

### Run the Main Algorithm

```bash
cd Deterministic-Matching-Policy
python Main.py
```

### Run Driver-Order Matching Demo

```bash
cd Deterministic_Matching
python Deterministic_Setting_Gurobi_Model_With_Fixed_Profit.py
```

---

## Technical Components

| Category | Details |
|----------|---------|
| **Mathematical Optimization** | Linear/Mixed-Integer Programming, Gurobi/CPLEX, Constraint formulation |
| **Stochastic Modeling** | Poisson processes, Scenario-based optimization, Monte Carlo simulation |
| **Machine Learning** | Reinforcement Learning (Q-Learning), Value function approximation, Softmax policies |
| **Algorithm Design** | Bipartite matching, Priority queues, Temporal decomposition |
| **Software Engineering** | Modular Python design, Statistical analysis, Experiment automation |

---

## Research Paper

This code implements the algorithms from:

**"Fleet Size Planning in Crowdsourced Delivery: Balancing Service Level and Driver Utilization"**

The paper addresses a novel research question at the intersection of:
- Operations Research (workforce scheduling, vehicle routing)
- Gig Economy (crowdsourced platforms, wage guarantees)
- Sequential Decision Making (multi-period optimization under uncertainty)

---

## Author

**Sahil Bhatt**  
Research Interests: Operations Research, Optimization, Machine Learning for Decision Making

---

## License

MIT License - See [LICENSE](LICENSE) for details
