import numpy as np
from typing import cast, Callable, TYPE_CHECKING
from agents.agent import Agent
from agents.producer import Producer
from agents.influencer import Influencer
if TYPE_CHECKING:
    from content_market import ContentMarket


class Consumer(Agent):
    def __init__(self, main_interest: np.ndarray, topic_interest_function: Callable[[float], float], attention_bound, external_interest_prob, delay_sensitivity):
        super().__init__(main_interest)

        self._producer_following_rates = dict()
        self._influencer_following_rates = dict()

        self._topic_interest_function = topic_interest_function
        self.attention_bound = attention_bound
        
        self.external_interest_prob = external_interest_prob
        self.delay_sensitivity = delay_sensitivity

    def init_following_rates(self):
        super().init_following_rates()
        cur_sum = 0
        for agent in self.market.agents:
            if agent == self or (not isinstance(agent, Producer) and not isinstance(agent, Influencer)):
                # only producers and influencers can be followed
                continue
            self._following_rates[agent.index] = np.random.uniform(0, self.attention_bound - cur_sum)
            cur_sum += self._following_rates[agent.index]
        
        self._following_rates['external'] = np.random.uniform(0, self.attention_bound - cur_sum)

    def consumption_topic_interest(self, topic: np.ndarray) -> float:
        if not self.market.check_topic(topic):
            raise ValueError("Topic is not in the market.")
        distance = np.linalg.norm(topic - self.main_interest)
        return self._topic_interest_function(distance)
    
    def check_following_rates(self, value: dict[int, float]) -> bool:
        if sum(value.values()) - self.attention_bound > 1e-6:
            return False
        return True

    def utility(self, x: np.array, *args) -> float:
        if self.market is None or self.index is None:
            raise ValueError("Consumer has no market.")
        production_rate: float = cast(float, args[0])
        external_production_rate: float = cast(float, args[1])

        following_rate_vector = x
        
        self.set_following_rate_vector(following_rate_vector)
        
        influencer_reward = 0
        for influencer in self.market.influencers:
            if self == influencer:
                continue
            for producer in self.market.producers:
                if not influencer.following_rates[producer.index] > 0:
                    continue
                if not self.following_rates[influencer.index] > 0:
                    continue

                if self == producer:
                    continue

                topic_reward = producer.topic_probability(producer.topic_produced) * self.consumption_topic_interest(producer.topic_produced)
                delay = np.exp(-self.delay_sensitivity * (1 / influencer.following_rates[producer.index] + 1 / self.following_rates[influencer.index]))

                influencer_reward += production_rate * topic_reward * delay
        
        direct_following_reward = 0
        for producer in self.market.producers:
            if not self.following_rates[producer.index] > 0:
                continue

            if self == producer:
                continue

            topic_reward = producer.topic_probability(producer.topic_produced) * self.consumption_topic_interest(producer.topic_produced)
            delay = np.exp(-self.delay_sensitivity * (1 / self.following_rates[producer.index]))
            direct_following_reward += production_rate * topic_reward * delay

        external_reward = 0
        if self.following_rates['external'] > 0:
            external_reward = external_production_rate * self.external_interest_prob * np.exp(-self.delay_sensitivity * (1 / self.following_rates['external']))

        return influencer_reward + direct_following_reward + external_reward