import numpy as np
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
if TYPE_CHECKING:
    from content_market import ContentMarket

class Agent(ABC):
    def __init__(self):
        self.market = None
        self.index = None
        self._following_rates = dict()

    def set_market(self, market: 'ContentMarket', index: int):
        self.market = market
        self.index = index

    def init_following_rates(self):
        """
        Initialize following rates based on the agent type.
        Override this method if you want to change the initialization.
        """
        self._following_rates = {agent.index: 0 for agent in self.market.agents}
        self._following_rates['external'] = 0

    @abstractmethod
    def reset(self):
        """
        Reset the agent to its initial state.
        """
        self.init_following_rates()

    def consumption_topic_interest(self, topic: np.ndarray) -> float:
        """
        Return the interest in a topic for consumption.
        """
        return 0

    def topic_probability(self, topic: np.ndarray) -> float:
        """
        Return the probability of producing a topic.
        """
        return 0

    def check_following_rates(self, value: dict[int, float]) -> bool:
        """
        Check if the following rates are valid.
        """
        return True

    @property
    def following_rates(self) -> dict[int, float]:
        return self._following_rates
    
    @following_rates.setter
    def following_rates(self, value: dict[int, float]):
        if not self.check_following_rates(value):
            raise ValueError("Following rates are not valid.")
        if value[self.index] > 0:
            raise ValueError("Agent cannot follow itself.")
        self._following_rates = value

    def get_following_rate_vector(self):
        return np.array(list(self.following_rates.values()))
    
    def set_following_rate_vector(self, following_rate_vector: np.array):
        if len(following_rate_vector) != len(self.following_rates):
            raise ValueError("Following rate vector has wrong length.")
        self.following_rates = {agent.index: following_rate_vector[agent.index] for agent in self.market.agents}
        self.following_rates['external'] = following_rate_vector[-1]

    @abstractmethod
    def utility(self, x: np.ndarray, *args) -> float:
        """
        Return the utility of the agent.
        x should be a vector of whatever input is needed for the utility function.
        For example, for a producer, x could be the topic produced.
        For a consumer, x could be the following rate vector.
        """
        pass

    def minimization_utility(self, x: np.ndarray, *args) -> float:
        """
        Return the utility of the agent.
        x should be a vector of whatever input is needed for the utility function.
        For example, for a producer, x could be the topic produced.
        For a consumer, x could be the following rate vector.
        """
        return -1 * self.utility(x, *args)