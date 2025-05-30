import unittest
import os
import numpy as np
import pandas as pd
import random
import copy
import sys
from unittest.mock import patch, MagicMock

script = os.path.dirname(__file__)
parent_dir = os.path.dirname(script)
# Set path to parent directory
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PPO_Environment import WNTRGymEnv
from mpl_toolkits.mplot3d import Axes3D

# filepath: test_PPO_Environment.py
import matplotlib.pyplot as plt

# Use absolute import since there's no __init__.py

class TestWNTRGymEnv(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment with sample data"""
        self.pipes = {
            'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
            'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
            'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
            'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
            'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
            'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
        }
        
        # Using a mock scenario for testing
        self.scenarios = ['test_scenario']
        
        # Create a mock environment
        with patch('os.path.join') as mock_join, \
             patch('os.listdir') as mock_listdir, \
             patch('wntr.network.WaterNetworkModel') as mock_wntr:
            
            # Mock the network states
            mock_join.return_value = 'mock_path'
            mock_listdir.return_value = ['network_0.inp', 'network_1.inp']
            
            # Mock the network model
            mock_network = MagicMock()
            mock_network.pipe_name_list = ['pipe1', 'pipe2', 'pipe3']
            mock_network.junction_name_list = ['j1', 'j2', 'j3']
            mock_wntr.return_value = mock_network
            
            # Create the environment
            self.env = WNTRGymEnv(self.pipes, self.scenarios, networks_folder='test_folder')
            
            # Mock the simulate_network method to avoid actual simulations
            self.env.simulate_network = MagicMock(return_value=(MagicMock(), {}))
            
    def test_initialization(self):
        """Test that environment initializes correctly"""
        self.assertEqual(self.env.pipe_IDs, list(self.pipes.keys()))
        self.assertEqual(len(self.env.pipe_diameter_options), len(self.pipes))
        self.assertEqual(self.env.action_space.n, len(self.pipes) + 1)
        
    @patch('wntr.network.WaterNetworkModel')
    @patch('os.listdir')
    @patch('os.path.join')
    def test_reset(self, mock_join, mock_listdir, mock_wntr):
        """Test reset method"""
        # Setup mocks
        mock_join.return_value = 'mock_path'
        mock_listdir.return_value = ['network_0.inp', 'network_1.inp']
        
        mock_network = MagicMock()
        mock_network.pipe_name_list = ['pipe1', 'pipe2', 'pipe3']
        mock_network.junction_name_list = ['j1', 'j2']
        mock_wntr.return_value = mock_network
        
        # Mock methods to avoid actual simulations
        self.env.simulate_network = MagicMock(return_value=(MagicMock(), {}))
        self.env.get_network_features = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        
        # Test reset
        obs = self.env.reset()
        
        # Assert state is reset
        self.assertEqual(self.env.current_time_step, 0)
        self.assertEqual(self.env.current_pipe_index, 0)
        self.assertIsNotNone(self.env.current_network)
        self.assertEqual(self.env.pipe_names, mock_network.pipe_name_list)
        
    @patch('wntr.network.WaterNetworkModel')
    @patch('os.listdir')
    @patch('os.path.join')
    def test_step(self, mock_join, mock_listdir, mock_wntr):
        """Test step method"""
        # Setup mocks similar to reset test
        mock_join.return_value = 'mock_path'
        mock_listdir.return_value = ['network_0.inp', 'network_1.inp']
        
        mock_network = MagicMock()
        mock_network.pipe_name_list = ['pipe1', 'pipe2']
        mock_network.junction_name_list = ['j1', 'j2']
        
        # Mock get_link to return pipes with diameters
        mock_pipe = MagicMock()
        mock_pipe.diameter = 0.5
        mock_network.get_link.return_value = mock_pipe
        
        # Mock to_graph to test disconnection checks
        mock_graph = MagicMock()
        mock_network.to_graph.return_value = mock_graph
        mock_graph.to_undirected.return_value = mock_graph
        
        mock_wntr.return_value = mock_network
        
        # Mock methods to avoid actual simulations
        self.env.simulate_network = MagicMock(return_value=(MagicMock(), {}))
        self.env.get_network_features = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        self.env.check_disconnections = MagicMock(return_value=(False, []))
        self.env.reward = MagicMock(return_value=(1.0, 100, 0.8, 0.9, False, [], False))
        
        # Reset the environment first
        self.env.reset()
        
        # Test step with action 0 (no change)
        obs, reward, done, info = self.env.step(0)
        
        # After one step, we should be on the next pipe
        self.assertEqual(self.env.current_pipe_index, 1)
        
        # Test step with action 1 (change diameter)
        obs, reward, done, info = self.env.step(1)
        
        # After this step, we've processed all pipes and should be on time step 1
        self.assertEqual(self.env.current_time_step, 1)
        self.assertEqual(self.env.current_pipe_index, 0)

def analyse_reward_vs_diameter_changes(scenario = 'anytown_densifying_3', num_simulations=50, plot=True):
    """
    Analyse how pipe diameter distribution changes affect rewards.
    
    Args:
        num_simulations: Number of simulations to run
        plot: Whether to generate plots
        
    Returns:
        DataFrame with analysis results
    """
    # Set up environment
    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }

    scenarios = [scenario]
    
    # Create environment
    env = WNTRGymEnv(pipes, scenarios)
    
    # Data collection
    results = []
    
    for sim in range(num_simulations):
        # Reset environment
        obs = env.reset()
        network = copy.deepcopy(env.current_network)
        
        # Store original diameters
        original_diameters = {}
        for pipe_name in env.pipe_names:
            pipe = network.get_link(pipe_name)
            original_diameters[pipe_name] = pipe.diameter
        
        # Generate random actions (either no change or upgrade)
        pipe_actions = []
        for i in range(len(env.pipe_names)):
            # Randomly choose action (0 = no change, 1-6 = change diameter)
            # Biased toward more frequent upgrades for this analysis
            action_prob = random.random()
            if action_prob < 0.7:  # 70% chance of upgrade
                action = random.randint(1, len(env.pipe_diameter_options))
            else:
                action = 0  # No change
                
            pipe_actions.append(action)
        
        # Apply actions and track changes
        num_upgrades = 0
        total_diameter_increase = 0
        diameter_changes = []
        
        for pipe_idx, action in enumerate(pipe_actions):
            # Process the action
            if action > 0:  # Change pipe diameter
                pipe_name = env.pipe_names[pipe_idx]
                pipe = env.current_network.get_link(pipe_name)
                old_diameter = pipe.diameter
                new_diameter = env.pipe_diameter_options[action - 1]
                
                # Only apply if it's an upgrade
                if new_diameter > old_diameter:
                    diameter_change = new_diameter - old_diameter
                    pipe.diameter = new_diameter
                    env.actions.append((pipe_name, new_diameter))
                    num_upgrades += 1
                    total_diameter_increase += diameter_change
                    diameter_changes.append((pipe_name, old_diameter, new_diameter, diameter_change))
            
            # Advance the pipe index
            env.current_pipe_index += 1
        
        # Run simulation after all pipes are processed
        results_sim, performance_metrics = env.simulate_network(env.current_network)
        
        # Calculate reward
        reward, cost, pd_ratio, demand_satisfaction, disconnections, actions_causing_disconnections, upgraded = env.reward(
            original_diameters,
            env.actions,
            performance_metrics,
            False,  # Not downgraded
            True if num_upgrades > 0 else False,  # Upgraded
            []
        )
        
        # Calculate metrics for diameter change distribution
        if diameter_changes:
            avg_diameter_change = sum(change[3] for change in diameter_changes) / len(diameter_changes)
            max_diameter_change = max(change[3] for change in diameter_changes) if diameter_changes else 0
            min_diameter_change = min(change[3] for change in diameter_changes) if diameter_changes else 0
            std_diameter_change = np.std([change[3] for change in diameter_changes]) if len(diameter_changes) > 1 else 0
        else:
            avg_diameter_change = max_diameter_change = min_diameter_change = std_diameter_change = 0
        
        # Calculate percentage of pipes upgraded
        pct_pipes_upgraded = (num_upgrades / len(env.pipe_names)) * 100
        
        # Store results
        results.append({
            'sim': sim,
            'reward': reward,
            'cost': cost,
            'pd_ratio': pd_ratio,
            'demand_satisfaction': demand_satisfaction,
            'num_upgrades': num_upgrades,
            'pct_pipes_upgraded': pct_pipes_upgraded,
            'total_diameter_increase': total_diameter_increase,
            'avg_diameter_change': avg_diameter_change,
            'max_diameter_change': max_diameter_change,
            'min_diameter_change': min_diameter_change,
            'std_diameter_change': std_diameter_change,
            'disconnections': disconnections
        })
        
        # Reset for next simulation
        env.current_pipe_index = 0
        env.actions = []
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if plot:
        plot_reward_analysis(results_df)
    
    return results_df

def plot_reward_analysis(results_df):
    """Generate plots to analyse reward vs pipe diameter changes"""
    plt.figure(figsize=(18, 12))
    plot_dir = os.path.join(parent_dir, 'Plots', 'Tests')
    os.makedirs(plot_dir, exist_ok=True)

    plt.suptitle(f'Reward vs Diameter Upgrade {scenario}', fontsize=16)
    
    # Plot 1: Reward vs Percentage of Pipes Upgraded
    plt.subplot(2, 3, 1)
    plt.scatter(results_df['pct_pipes_upgraded'], results_df['reward'], alpha=0.7)
    plt.xlabel('Percentage of Pipes Upgraded')
    plt.ylabel('Reward')
    plt.title('Reward vs Percentage of Pipes Upgraded')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Reward vs Total Diameter Increase
    plt.subplot(2, 3, 2)
    plt.scatter(results_df['total_diameter_increase'], results_df['reward'], alpha=0.7)
    plt.xlabel('Total Diameter Increase (m)')
    plt.ylabel('Reward')
    plt.title('Reward vs Total Diameter Increase')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Reward vs Average Diameter Change
    plt.subplot(2, 3, 3)
    plt.scatter(results_df['avg_diameter_change'], results_df['reward'], alpha=0.7)
    plt.xlabel('Average Diameter Change (m)')
    plt.ylabel('Reward')
    plt.title('Reward vs Average Diameter Change')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Reward vs Standard Deviation of Diameter Changes
    plt.subplot(2, 3, 4)
    plt.scatter(results_df['std_diameter_change'], results_df['reward'], alpha=0.7)
    plt.xlabel('Std Dev of Diameter Changes')
    plt.ylabel('Reward')
    plt.title('Reward vs Distribution of Diameter Changes')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 5: Reward vs Cost
    plt.subplot(2, 3, 5)
    plt.scatter(results_df['cost'], results_df['reward'], alpha=0.7)
    plt.xlabel('Cost')
    plt.ylabel('Reward')
    plt.title('Reward vs Cost')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 6: Reward vs Pressure-Demand Ratio
    plt.subplot(2, 3, 6)
    plt.scatter(results_df['pd_ratio'], results_df['reward'], alpha=0.7)
    plt.xlabel('Pressure-Demand Ratio')
    plt.ylabel('Reward')
    plt.title('Reward vs Pressure-Demand Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'reward_vs_diameter_upgrades.png'), dpi=300)
    plt.show()
    
    # Additional plots for deeper analysis
    plt.figure(figsize=(15, 10))
    # Add parent title of the scenario
    plt.suptitle(f'Upgrade Details Analysis for {scenario}', fontsize=16)
    
    # Plot histogram of rewards
    plt.subplot(2, 2, 1)
    plt.hist(results_df['reward'], bins=20, alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rewards')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot diameter increase vs demand satisfaction
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['total_diameter_increase'], results_df['demand_satisfaction'], alpha=0.7)
    plt.xlabel('Total Diameter Increase (m)')
    plt.ylabel('Demand Satisfaction')
    plt.title('Diameter Increase vs Demand Satisfaction')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3D scatter of key factors
    if results_df.shape[0] > 10:  # Only if we have enough data points
        ax = plt.subplot(2, 2, 3, projection='3d')
        ax.scatter(
            results_df['total_diameter_increase'], 
            results_df['pd_ratio'], 
            results_df['reward'],
            c=results_df['reward'], 
            cmap='viridis',
            alpha=0.7
        )
        ax.set_xlabel('Total Diameter Increase')
        ax.set_ylabel('PD Ratio')
        ax.set_zlabel('Reward')
        ax.set_title('3D Relationship: Diameter Increase, PD Ratio, and Reward')
    
    # Plot correlation heatmap
    plt.subplot(2, 2, 4)
    corr = results_df.corr()
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'upgrade_details_{scenario}.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    # Run the test suite
    unittest.main(exit=False)
    
    print("\nRunning reward vs diameter changes analysis...")
    # analyse how pipe diameter changes affect rewards

    scenarios = ['anytown_densifying_3', 'hanoi_densifying_3', 'anytown_sprawling_3', 'hanoi_sprawling_3']

    for scenario in scenarios:
        print(f"Analyzing scenario: {scenario}")
        results_df = analyse_reward_vs_diameter_changes(num_simulations=50)
        
        # Save results to CSV for further analysis
        results_df.to_csv(os.path.join(script, f'reward_vs_diameter_{scenario}.csv'), index=False)
        print(f"Analysis complete. Results saved to CSV and plots generated.")