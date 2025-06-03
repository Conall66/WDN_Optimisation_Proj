# WDN Optimisation project

## Introduction

This project uses deep reinforcement learning (PPO) to optimise the pipe sizing of different evolving water distribution networks. By implementing a GNN architecture for the actor-critic networks, the agent can handle rapidly sprawling networks.
In order to train the agent, first upload a folder a particular network and various scenarios for how the evolution might play out. Run this scenario through the Train_w_Plots script to see the agent stored in the 'agents' folder. You can observe the performance metrics of the agent, reward maximisation against random policy selection, and how ecah of the components of the reward function are tailored throughout training. 
