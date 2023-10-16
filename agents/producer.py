from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    from content_market import ContentMarket

import numpy as np

class Producer:

    def __init__(self, index: int, main_interest: np.ndarray, topic_interest_function):
        self.market = None
        self.main_interest = main_interest
        self.topic_produced = main_interest
        
        self._topic_interest_function = topic_interest_function
        self.index = index

    def set_market(self, market: 'ContentMarket'):
        self.market = market
        if not market.check_topic(self.main_interest):
            raise ValueError("Main interest is not in the market.")

    def topic_probability(self, topic: np.ndarray) -> float:
        if not self.market.check_topic(topic):
            raise ValueError("Topic is not in the market.")
        distance = np.linalg.norm(topic - self.main_interest)
        return self._topic_interest_function(distance)

    def utility(self, topic: np.ndarray, *args) -> float:
        if self.market is None:
            raise ValueError("Producer has no market.")
        production_rate = cast(float, args[0])
        
        influencer_reward = 0
        for influencer in self.market.influencers:
            if not influencer.producer_following_rates[self.index] > 0:
                continue
            if influencer == self:
                continue
            for consumer in self.market.consumers:
                if not consumer.influencer_following_rates[influencer.index] > 0:
                    continue
                if consumer == self:
                    continue

                consumer_interest = self.topic_probability(topic) * consumer.consumption_topic_interest(topic)
                delay = np.exp(-influencer.delay_sensitivity * (1 / influencer.producer_following_rates[self.index] + 1 / consumer.influencer_following_rates[influencer.index]))

                influencer_reward += production_rate * consumer_interest * delay

        direct_consumer_reward = 0
        for consumer in self.market.consumers:
            if not consumer.producer_following_rates[self.index] > 0:
                continue
            if consumer == self:
                continue

            consumer_interest = self.topic_probability(topic) * consumer.consumption_topic_interest(topic)
            delay = np.exp(-consumer.delay_sensitivity * (1 / consumer.producer_following_rates[self.index]))

            direct_consumer_reward += production_rate * consumer_interest * delay

        return influencer_reward + direct_consumer_reward
    
    def minimization_utility(self, topic: np.ndarray, *args) -> float:
        return -1 * self.utility(topic, *args)