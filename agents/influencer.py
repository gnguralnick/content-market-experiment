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
            if self.market.producers[i] == self:
                self._producer_following_rates[i] = 0
            else:
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

    def utility(self, following_rate_vector: np.ndarray, *args) -> float:
        if self.market is None:
            raise ValueError("Influencer has no market.")
        production_rate = cast(float, args[0])
        
        self.set_following_rate_vector(following_rate_vector)
        
        reward = 0
        for consumer in self.market.consumers:
            if not consumer.influencer_following_rates[self.index] > 0:
                continue
            if consumer == self:
                continue
            for producer in self.market.producers:
                
                if not self.producer_following_rates[producer.index] > 0:
                    continue
                if consumer == producer:
                    continue
                if producer == self:
                    continue

                consumer_interest = producer.topic_probability(producer.topic_produced) * consumer.consumption_topic_interest(producer.topic_produced)
                delay = np.exp(-self.delay_sensitivity * (1 / self.producer_following_rates[producer.index] + 1 / consumer.influencer_following_rates[self.index]))

                reward += production_rate * consumer_interest * delay

        return reward

    def minimization_utility(self, following_rate_vector: np.ndarray, *args) -> float:
        return -1 * self.utility(following_rate_vector, *args)