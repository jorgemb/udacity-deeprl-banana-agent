# Agent to solve the Banana problem
from abc import abstractmethod
from os import urandom
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DQNAgent import DQNAgent
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

class Agent:
    def __init__(self, state_size, action_size, gamma=1.0, alpha=0.1, seed=-1) -> None:
        """ Initializes the Agent with environment information and hyperparameters

        Args:
            space_size (int): Size of the state space
            action_size (int): Size of the action space
            gamma (float, optional): Discount rate. Defaults to 1.0.
            alpha (float, optional): Learning rate. Defaults to 0.1.
            seed (int, optional): Seed. If -1 then a random seed is used.
        """
        self.space_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.alpha = alpha

        self.epsilon = 1.0
        self.epsilon_min = 0.05

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        if(seed != -1):
            random.seed(seed)
        else:
            seed = int.from_bytes(urandom(4), byteorder="little")
            random.seed(seed)
        self.seed = seed


    @abstractmethod
    def start(self, state):
        """Marks the start of an episode with the initial state. Returns the initial action.

        Args:
            state (array_like): Initial state

        Returns:
            int: Initial action
        """
        pass

    @abstractmethod
    def step(self, reward, state):
        """Performs a step in the simulation, provides the reward of the last action and the next state.

        Args:
            reward (float): Reward from previous action
            state (array_like): New state

        Returns:
            int: Next action
        """
        pass

    @abstractmethod
    def end(self, reward):
        """Finishes an episode, provides the last reward that was provided.

        Args:
            reward (float): Reward from previous action
        """
        pass

    @abstractmethod
    def __getstate__(self):
        """Return a state for pickling
        """
        pass

    @abstractmethod
    def __setstate__(self, state):
        """Restores the instance attributes from a state

        Args:
            state (dict): Dictionary of elements
        """
        pass

    def agent_name(self):
        return self.__class__.__name__

class Udagent(Agent):
    def __init__(self, state_size, action_size, gamma=1, alpha=0.1, seed=-1) -> None:
        super().__init__(state_size, action_size, gamma, alpha, seed)

        self.agent = DQNAgent(state_size, action_size, self.seed, QModel)
        self.last_state = None
        self.last_action = None
        
    def start(self, state):
        self.last_state = state
        self.last_action = self.agent.act(state, self.epsilon)

        return self.last_action

    def step(self, reward, state):
        self.agent.step(self.last_state, self.last_action, reward, state, False)
        self.last_action = self.agent.act(state, self.epsilon)
        return self.last_action

    def end(self, reward):
        self.agent.step(self.last_state, self.last_action, reward, self.last_state, True)
        self.epsilon = max(self.epsilon * 0.999, self.epsilon_min)
    
    def __getstate__(self):
        # Ignore the memory data
        state = self.__dict__.copy()
        state["agent"].memory = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.agent.memory = DQNAgent.create_memory(self.action_size, self.seed)


class BananaAgent(Agent):
    """Agent for traversing through the Banana environment using vanilla DQN with MemoryReplay
    """
    def __init__(self, state_size, action_size, gamma=1, alpha=0.1, seed=-1, tau=1e-3, buffer_size=int(1e5), batch_size=64, learn_every=4) -> None:
        super().__init__(state_size, action_size, gamma, alpha, seed)
    
        self.last_state = None
        self.last_action = None

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
    def __init__(self, state_size, action_size, gamma=1, alpha=0.1, seed=-1, tau=0.001, buffer_size=int(100000), batch_size=64, learn_every=4) -> None:
        super().__init__(state_size, action_size, gamma, alpha, seed, tau, buffer_size, batch_size, learn_every)
        
    def learning_step(self):
        # Check that memory has enough data
        if len(self.memory) < self.batch_size:
            return
        
        # Get buffer sample
        s, a, r, s_i, dones = self.memory.sample()

        # Predict using target network
        with torch.no_grad():
            # Q_argmax = self.q_local(s_i).detach().argmax(1).unsqueeze(1)
            # Q_max = self.q_target(s_i).detach().gather(1, Q_argmax)
            Q_argmax = self.q_local(s_i).argmax(1).unsqueeze(1)
            Q_max = self.q_target(s_i).gather(1, Q_argmax)
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
