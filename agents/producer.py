from typing import cast, TYPE_CHECKING
from agents.agent import Agent
if TYPE_CHECKING:
    from content_market import ContentMarket

import numpy as np

class Producer(Agent):

    def __init__(self, topic_interest_function):
        Agent.__init__(self)

        self.main_interest = None
        self.topic_produced = None
        self._topic_interest_function = topic_interest_function

    def set_main_interest(self, main_interest: np.ndarray):
        self.main_interest = main_interest

    def reset(self):
        self.topic_produced = self.main_interest

    def topic_probability(self, topic: np.ndarray) -> float:
        if not self.market.check_topic(topic):
            raise ValueError("Topic is not in the market.")
        distance = np.linalg.norm(topic - self.main_interest)
        return self._topic_interest_function(distance)

    def utility(self, x: np.ndarray, *args) -> float:
        if self.market is None or self.index is None:
            raise ValueError("Producer has no market.")
        production_rate = cast(float, args[0])

        topic = x
        
        influencer_reward = 0
        for influencer in self.market.influencers:
            if not influencer.following_rates[self.index] > 0:
                continue
            if influencer == self:
                continue
            for consumer in self.market.consumers:
                if not consumer.following_rates[influencer.index] > 0:
                    continue
                if consumer == self:
                    continue

                consumer_interest = self.topic_probability(topic) * consumer.consumption_topic_interest(topic)
                delay = np.exp(-influencer.delay_sensitivity * (1 / influencer.following_rates[self.index] + 1 / consumer.following_rates[influencer.index]))

                influencer_reward += production_rate * consumer_interest * delay

        direct_consumer_reward = 0
        for consumer in self.market.consumers:
            if not consumer.following_rates[self.index] > 0:
                continue
            if consumer == self:
                continue

            consumer_interest = self.topic_probability(topic) * consumer.consumption_topic_interest(topic)
            delay = np.exp(-consumer.delay_sensitivity * (1 / consumer.following_rates[self.index]))

            direct_consumer_reward += production_rate * consumer_interest * delay

        return influencer_reward + direct_consumer_reward