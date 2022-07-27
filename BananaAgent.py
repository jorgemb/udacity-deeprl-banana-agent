# Agent to solve the Banana problem
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Agent import Agent
from ReplayBuffer import ReplayBuffer


class QModel(nn.Module):
    """Model to use for Q-Learning
    """
    def __init__(self, space_size, action_size, seed):
        super().__init__()
        torch.manual_seed(seed)

        # Network
        h1 = 64
        h2 = 64
        self.hl1 = nn.Linear(space_size, h1)
        self.hl2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, action_size)
    
    def forward(self, x):
        x = self.hl1(x)
        x = F.relu(x)
        x = self.hl2(x)
        x = F.relu(x)
        x = self.out(x)
        
        return x


class BananaAgent(Agent):
    """Agent for traversing through the Banana environment using vanilla DQN with MemoryReplay
    """
    def __init__(self, state_size, action_size, *, 
                 gamma=1, alpha=0.1, seed=-1, tau=1e-3, buffer_size=int(1e5), batch_size=64, learn_every=4, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.steps = 0
        self.learn_every = learn_every

        # Model to use for learning
        self.q_local = QModel(state_size, action_size, self.seed).to(self.device)
        self.q_target = QModel(state_size, action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=alpha)
        
        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device, self.seed)
        
        # Last states and actions
        self.last_state = None
        self.last_action = None
    
    def start(self, state):
        self.last_state = state
        self.last_action = self.e_greedy_action(self.last_state, self.epsilon)

        return self.last_action
    
    def e_greedy_action(self, state, epsilon):
        """Take a greedy action according to the given state.

        Args:
            state (array_like): Current state to use
            epsilon (float): Epsilon to use for greedy action

        Returns:
            _type_: _description_
        """
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # Action selection according to epsilon
        if random.random() > epsilon:
            # Evaluate network
            self.q_local.eval()
            with torch.no_grad():
                action_values = self.q_local(x)
            self.q_local.train()

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learning_step(self):
        """Perform a learning step with memory buffer experiences
        """
        # Check that memory has enough data
        if len(self.memory) < self.batch_size:
            return
        
        # Get buffer sample
        s, a, r, s_i, dones = self.memory.sample()

        # Predict using target network
        Q_max = self.q_target(s_i).detach().max(1)[0].unsqueeze(1)
        target = r + self.gamma * Q_max * (1 - dones)
        
        # Get expected values with local network
        expected = self.q_local(s).gather(1, a)
        
        # Compute loss and optimize
        loss = F.mse_loss(expected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        # Update target network
        for target_param, local_param in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau)*target_param.data)

    def step(self, reward, state):
        # Add experience
        self.memory.add(self.last_state, self.last_action, reward, state, False)
        self.steps += 1

        # See if the agent should update
        if self.steps % self.learn_every == 0:
            # Learning step
            self.learning_step()
       
        # Save last action and state
        self.last_state = state
        self.last_action = self.e_greedy_action(state, self.epsilon)

        return self.last_action

    def end(self, reward):
        # Add experience
        self.memory.add(self.last_state, self.last_action, reward, self.last_state, True)
        
        # Update epsilon
        self.epsilon = max(self.epsilon * 0.999, self.epsilon_min)

    def __getstate__(self):
        # Ignore the memory data
        state = self.__dict__.copy()
        del state["memory"]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device, self.seed)

class BananaAgentDouble(BananaAgent):
    """Implements a BananaAgent using Double Q-Learning
    """
    def __init__(self, state_size, action_size, **kwargs) -> None:
        super().__init__(state_size, action_size, **kwargs)
        
    def learning_step(self):
        # Check that memory has enough data
        if len(self.memory) < self.batch_size:
            return
        
        # Get buffer sample
        s, a, r, s_i, dones = self.memory.sample()

        # Predict using target network
        Q_argmax = self.q_local(s_i).detach().argmax(1).unsqueeze(1)
        Q_max = self.q_target(s_i).detach().gather(1, Q_argmax)
        target = r + self.gamma * Q_max * (1 - dones)
        
        # Get expected values with local network
        expected = self.q_local(s).gather(1, a)
        
        # Compute loss and optimize
        loss = F.mse_loss(expected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        for target_param, local_param in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau)*target_param.data)
