#!/usr/bin/env python
# coding: utf-8

import os
import time
from collections import deque

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

import BananaAgent as Agents
import wandb

def do_episode(environment, agent):
    """Performs a single episode using the given environment and agent

    Args:
        environment (env): Environment that will perform the simulation
        agent (Agent): Agent that will traverse the environment

    Returns:
        (float, int): Total score and steps of the episode
    """
    episode_score = 0
    env_info = env.reset(train_mode=True)[brain_name]

    # Start the agent
    state = env_info.vector_observations[0]
    next_action = agent.start(state)

    # Take the first action
    env_info = env.step(next_action)[brain_name]

    while not env_info.local_done[0]:
        # Take a step from the agent
        reward = env_info.rewards[0]
        episode_score += reward
        state = env_info.vector_observations[0]

        next_action = agent.step(reward, state)

        # Perform action
        env_info = env.step(next_action)[brain_name]
    
    # Register last reward to the agent
    reward = env_info.rewards[0]
    episode_score += reward
    agent.end(reward)

    return episode_score


def create_agent(state_space, action_space, **kwargs):
    """Create the list of agents to test

    Returns:
        list: List of agents
    """
    agent = kwargs.get('agent')
    if agent == 'DQN':
        return Agents.BananaAgent(state_space, action_space, **kwargs)
    elif agent == 'DoubleDQN':
        return Agents.BananaAgentDouble(state_space, action_space, **kwargs)
    else:
        print(f"No agent named: {agent}")
        return None


    
def do_experiment(environment, agent, episodes, print_every):
    """Performs an experiment on the given agent.

    Args:
        environment (any): Environment to use
        agent (Agent): Agent that follows the "Agent" interface
        episodes (int): Amount of episodes to perform
        print_every (int): How often to print the episode information

    Returns:
        (array_like, array_like): Scores and times that the agent took per episode
    """
    scores = np.zeros(episodes)
    times = np.zeros(episodes)

    for i in range(episodes):
        start_time = time.time()
        scores[i] = do_episode(environment, agent)
        times[i] = time.time() - start_time

        # Log data
        ep = i+1
        wandb.log({
            "episode": ep,
            "score": scores[i],
            "time": times[i]
        })
        
        if ep % print_every == 0:
            print(f"{agent.agent_name()} :: ({ep}/{episodes}) AVG {np.average(scores[max(0, i-print_every):])}")
    
    return scores, times

if __name__ == '__main__':
    # TODO: Check if Banana is available
    # Load environment and get initial brain
    env = UnityEnvironment("Banana/Banana.x86_64", no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Initialize environment for use of the agent
    env_info = env.reset(train_mode=True)[brain_name]
    action_space = brain.vector_action_space_size
    state_space = env_info.vector_observations.size

    with wandb.init(project='nanorl-p1') as run:
        average_window = wandb.config.avg_window
        episodes = wandb.config.episodes
        print_every = 400

        agent = create_agent(state_space, action_space, **wandb.config)
        # Do experiment
        scores, times = do_experiment(env, agent, episodes, print_every)
        agent_name = agent.agent_name()

        # Save agent
        torch.save(agent, f"agents/{agent_name}-{run.id}.pt")

    # Close environment
    env.close()

