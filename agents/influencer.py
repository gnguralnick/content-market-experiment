from typing import cast, TYPE_CHECKING
from agents.agent import Agent
from agents.producer import Producer
if TYPE_CHECKING:
    from content_market import ContentMarket

import numpy as np

class Influencer(Agent):

    def __init__(self, attention_bound, delay_sensitivity, init_following_rates_method: str = 'random'):
        Agent.__init__(self)
        
        self.attention_bound = attention_bound
        self.delay_sensitivity = delay_sensitivity

        self.init_following_rates_method = init_following_rates_method

    def init_following_rates(self):
        Agent.init_following_rates(self)
        if self.init_following_rates_method == 'random':
            cur_sum = 0
            agent_random_sort = np.random.permutation(self.market.agents)
            for agent in agent_random_sort:
                if agent == self or not isinstance(agent, Producer):
                    # only producers can be followed
                    continue
                self._following_rates[agent.index] = np.random.uniform(0, self.attention_bound - cur_sum)
                cur_sum += self._following_rates[agent.index]
        elif self.init_following_rates_method == 'equal':
            num_producers = len(set(self.market.producers) - {self})
            for agent in self.market.agents:
                if agent == self or not isinstance(agent, Producer):
                    # only producers and influencers can be followed
                    continue
                self._following_rates[agent.index] = self.attention_bound / num_producers
        elif self.init_following_rates_method == 'zero':
            pass
        else:
            raise ValueError("Unknown init_following_rates_method.")

    def reset(self):
        return Agent.reset(self)
    
    def get_following_rate_bounds(self):
        bounds = []
        for agent in self.market.agents:
            if agent == self:
                bounds.append((0, 0))
            elif not isinstance(agent, Producer):
                bounds.append((self.following_rates[agent.index], self.following_rates[agent.index]))
            else:
                bounds.append((0, None))
        bounds.append((self.following_rates['external'], self.following_rates['external'])) # external should not change when optimizing influencer
        return bounds

    def utility(self, x: np.ndarray, *args) -> float:
        if self.market is None or self.index is None:
            raise ValueError("Influencer has no market.")
        production_rate = cast(float, args[0])

        following_rate_vector = x
        
        reward = 0
        for consumer in self.market.consumers:
            if not consumer.following_rates[self.index] > 0:
                continue
            if consumer == self:
                continue
            for producer in self.market.producers:
                
                if not following_rate_vector[producer.index] > 0:
                    continue
                if consumer == producer:
                    continue
                if producer == self:
                    continue

                consumer_interest = producer.topic_probability(producer.topic_produced) * consumer.consumption_topic_interest(producer.topic_produced)
                delay = np.exp(-self.delay_sensitivity * (1 / following_rate_vector[producer.index] + 1 / consumer.following_rates[self.index]))

                reward += production_rate * consumer_interest * delay
        return reward