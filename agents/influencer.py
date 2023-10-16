from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    from content_market import ContentMarket

import numpy as np

class Influencer:

    def __init__(self, index: int, main_interest: np.ndarray, attention_bound, delay_sensitivity):
        self.market = None
        self.main_interest = main_interest
        
        self.attention_bound = attention_bound
        self.index = index
        self._producer_following_rates = dict()
        self.delay_sensitivity = delay_sensitivity

    def set_market(self, market: 'ContentMarket'):
        self.market = market
        if not market.check_topic(self.main_interest):
            raise ValueError("Main interest is not in the market.")
        
        self._producer_following_rates = {i: 0 for i in range(market.num_producers)}

        cur_sum = 0
        for i in range(market.num_producers):
            self._producer_following_rates[i] = np.random.uniform(0, self.attention_bound - cur_sum)
            cur_sum += self._producer_following_rates[i]

        # num_follows = self.market.num_producers
        # rate_per_follow = self.attention_bound / num_follows

        # self._producer_following_rates = {i: rate_per_follow for i in range(market.num_producers)}

    @property
    def producer_following_rates(self) -> dict[int, float]:
        return self._producer_following_rates
    
    @producer_following_rates.setter
    def producer_following_rates(self, value):
        if sum(value.values()) -  self.attention_bound > 1e-6:
            raise ValueError("Sum of following rates exceeds attention bound.")
        self._producer_following_rates = value

    def get_following_rate_vector(self):
        return np.array([self.producer_following_rates[i] for i in range(self.market.num_producers)])
    
    def set_following_rate_vector(self, following_rate_vector: np.array):
        if len(following_rate_vector) != self.market.num_producers:
            raise ValueError("Following rate vector has wrong length.")
        if sum(following_rate_vector) - self.attention_bound > 1e-6:
            raise ValueError("Sum of following rates exceeds attention bound.")
        self._producer_following_rates = {i: following_rate_vector[i] for i in range(self.market.num_producers)}

    @staticmethod
    def utility(following_rate_vector: np.ndarray, *args) -> float:
        influencer = cast(Influencer, args[0])
        if influencer.market is None:
            raise ValueError("Influencer has no market.")
        production_rate = cast(float, args[1])
        
        influencer.set_following_rate_vector(following_rate_vector)
        
        reward = 0
        for consumer in influencer.market.consumers:
            #print(consumer.influencer_following_rates)
            for producer in influencer.market.producers:
                if not consumer.influencer_following_rates[influencer.index] > 0:
                    continue
                if not influencer.producer_following_rates[producer.index] > 0:
                    continue
                if consumer == producer:
                    continue

                consumer_interest = producer.topic_probability(producer.topic_produced) * consumer.consumption_topic_interest(producer.topic_produced)
                delay = np.exp(-influencer.delay_sensitivity * (1 / influencer.producer_following_rates[producer.index] + 1 / consumer.influencer_following_rates[influencer.index]))

                reward += production_rate * consumer_interest * delay

        return reward

    @staticmethod
    def minimization_utility(following_rate_vector: np.ndarray, *args) -> float:
        return -1 * Influencer.utility(following_rate_vector, *args)