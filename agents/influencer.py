from typing import cast, TYPE_CHECKING
from agents.agent import Agent
from agents.producer import Producer
if TYPE_CHECKING:
    from content_market import ContentMarket

import numpy as np

class Influencer(Agent):

    def __init__(self, main_interest: np.ndarray, attention_bound, delay_sensitivity):
        super().__init__(main_interest)
        
        self.attention_bound = attention_bound
        self.delay_sensitivity = delay_sensitivity

    def init_following_rates(self):
        super().init_following_rates()
        cur_sum = 0
        for agent in self.market.agents:
            if agent == self or not isinstance(agent, Producer):
                # only producers can be followed
                continue
            self._following_rates[agent.index] = np.random.uniform(0, self.attention_bound - cur_sum)
            cur_sum += self._following_rates[agent.index]

    def utility(self, x: np.ndarray, *args) -> float:
        if self.market is None or self.index is None:
            raise ValueError("Influencer has no market.")
        production_rate = cast(float, args[0])

        following_rate_vector = x
        
        self.set_following_rate_vector(following_rate_vector)
        
        reward = 0
        for consumer in self.market.consumers:
            if not consumer.following_rates[self.index] > 0:
                continue
            if consumer == self:
                continue
            for producer in self.market.producers:
                
                if not self.following_rates[producer.index] > 0:
                    continue
                if consumer == producer:
                    continue
                if producer == self:
                    continue

                consumer_interest = producer.topic_probability(producer.topic_produced) * consumer.consumption_topic_interest(producer.topic_produced)
                delay = np.exp(-self.delay_sensitivity * (1 / self.following_rates[producer.index] + 1 / consumer.following_rates[self.index]))

                reward += production_rate * consumer_interest * delay

        return reward

    def minimization_utility(self, following_rate_vector: np.ndarray, *args) -> float:
        return -1 * self.utility(following_rate_vector, *args)