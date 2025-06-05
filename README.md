# GNN-DRL for Water Distribution Network Pipe Sizing Optimization

## Overview

This project implements a Deep Reinforcement Learning (DRL) agent, utilizing Graph Neural Networks (GNNs) and Proximal Policy Optimization (PPO), for optimizing pipe sizing and upgrades in Water Distribution Networks (WDNs). The system is designed to handle variable network topologies and evolving scenarios (e.g., densifying or sprawling demands) over a simulated time horizon. It integrates with the Water Network Tool for Resilience (WNTR) for hydraulic simulations and aims to make cost-effective intervention decisions while maintaining network performance (e.g., minimizing pressure deficit, ensuring demand satisfaction) under budget constraints.

The project also includes functionality for evaluating agent performance, visualizing training progress, and comparing against baseline policies. A Genetic Algorithm (GA) approach is also present (in `GA_Alt_Approach.py`) for alternative optimization or comparison.

## Key Features

* **DRL-based Pipe Sizing:** Utilizes PPO to learn optimal pipe upgrade strategies.
* **Graph Neural Networks:** Employs GNNs (specifically GCN-based architecture in `Actor_Critic_Nets2.py`) to process graph-structured WDN data, allowing for generalization across different network sizes and topologies.
* **WNTR Integration:** Leverages WNTR for realistic hydraulic simulations of WDNs using the EPANET engine.
* **Custom Gymnasium Environment:** `WNTRGymEnv` provides a custom environment where the DRL agent interacts with WDNs, taking actions (pipe upgrades) and receiving observations (network state as graph features) and rewards.
* **Dynamic Scenarios:** Designed to train and evaluate on multiple evolving WDN scenarios, including "Anytown" and "Hanoi" networks with densifying and sprawling demand patterns over 50 time steps.
* **Budget-Constrained Optimization:** The environment incorporates a budget mechanism, where the agent receives an initial budget and budget per step, and is penalized for exceeding available funds.
* **Comprehensive Reward Function:** The reward considers intervention costs, pressure deficit, demand satisfaction, and penalties for exceeding the budget.
* **Monitoring & Visualization:** Includes a `PlottingCallback` and various functions in `Plot_Agents.py` to log training metrics, visualize agent performance, training progress, action analysis, and scenario-specific results.
* **Genetic Algorithm:** `GA_Alt_Approach.py` (details specific to its implementation would be in that file) provides a GA-based optimization approach.

## Core Components / File Structure

* `PPO_Environment.py`: Defines `WNTRGymEnv`, the custom Gymnasium environment for WDN interactions. It handles network loading, state representation (graph features for GNN), action application, hydraulic simulation calls, and reward calculation.
* `Actor_Critic_Nets2.py`: Implements the `GNNFeatureExtractor` which processes raw graph observations into feature vectors for the policy. It also defines the `GNNActorCriticPolicy` and the `GraphPPOAgent` which wraps the Stable Baselines3 PPO algorithm.
* `Reward.py`: Contains functions to calculate the reward signal based on network performance, intervention costs, and budget adherence.
* `Hydraulic_Model.py`: A wrapper for running WNTR (EPANET) simulations and evaluating network performance metrics like pressure deficit and demand satisfaction.
* `Train_w_Plots.py`: The main script for orchestrating DRL agent training, including different training regimes (e.g., specific networks, combined networks), agent evaluation, and calling plotting functions.
* `Plot_Agents.py`: Provides the `PlottingCallback` for logging metrics during SB3 training and various functions to generate plots for training progress, step-wise performance, action analysis, and scenario comparisons.
* `GA_Alt_Approach.py`: Implements a Genetic Algorithm for WDN optimization.
* `Visualise_network.py`: Utilities for visualizing network properties, such as pipe diameter heatmaps over time (used by `Train_w_Plots.py`).
* `Modified_nets/`: Directory containing the WDN scenario input files (`.inp`), organized by scenario type and step.
* `agents/`: Default directory for saving trained DRL models.
* `Plots/`: Default directory for saving training logs (`.csv`) and generated plots.

## Requirements

* Python 3.x
* `torch`
* `torch_geometric`
* `stable-baselines3`
* `sb3_contrib` (especially if MaskablePPO variants are explored)
* `wntr`
* `numpy`
* `pandas`
* `matplotlib`
* `gymnasium` (used in `PPO_Environment.py` and `Actor_Critic_Nets2.py` explicitly)
* `networkx` (a dependency of WNTR and PyTorch Geometric)

An EPANET dynamic link library (DLL) or shared object (SO) compatible with WNTR is also required for hydraulic simulations.

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    Create a `requirements.txt` file based on the libraries listed above and install using:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install them individually:
    ```bash
    pip install torch torch_geometric stable-baselines3 sb3_contrib wntr numpy pandas matplotlib gymnasium networkx
    ```
4.  **Ensure EPANET is accessible:** WNTR needs access to the EPANET solver. Ensure it's installed and correctly configured in your system path or WNTR settings.

## Usage

### Training a DRL Agent

The primary script for training is `Train_w_Plots.py`. It contains several functions to initiate training runs:

* `train_just_anytown()`: Trains an agent solely on Anytown network scenarios.
* `train_just_hanoi(single_scenario=False)`: Trains an agent solely on Hanoi network scenarios. Set `single_scenario=True` to train on only one specific Hanoi scenario (currently `hanoi_sprawling_1`).
* `train_both()`: Trains an agent on a combination of Anytown and Hanoi scenarios (currently configured to use Hanoi scenarios for the `DummyVecEnv` in this function).
* `train_multiple()`: An example function to train multiple agents with different configurations (e.g., Anytown only, Hanoi only, sequential training).

To run a training session, uncomment the desired function call at the end of `Train_w_Plots.py` and execute:
```bash
python Train_w_Plots.py
