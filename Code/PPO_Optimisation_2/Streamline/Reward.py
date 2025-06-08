
"""

In this script, we design a reward function for training the RL agent in the PPO environment. The reward function is designed to encourage the agent to optimize the water distribution network by minimising costs while maintaining hydraulic performance.

"""

# First reward function aims to simply minimse costs. We should see the agent makes no changes to the network at all, since all changes have an incurred cost.

def reward_function_cost_only(wn):

    """
    Calculate the reward based on the cost of any pipe changes
    """

