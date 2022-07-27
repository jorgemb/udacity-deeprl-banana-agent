from abc import abstractmethod
from os import urandom
import torch
import random

class Agent:
    def __init__(self, state_size, action_size, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        """ Initializes the Agent with environment information and hyperparameters

        Args:
            space_size (int): Size of the state space
            action_size (int): Size of the action space
            gamma (float, optional): Discount rate. Defaults to 1.0.
            alpha (float, optional): Learning rate. Defaults to 0.1.
            seed (int, optional): Seed. If -1 then a random seed is used.
        """
        self.state_size = state_size
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
    def step(self, reward, state, learn=True):
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

