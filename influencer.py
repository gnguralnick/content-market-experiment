import numpy as np
from content_market import ContentMarket

class Influencer:

    def __init__(self, market: ContentMarket, main_interest: np.ndarray, attention_bound, index, delay_sensitivity):
        self.market = market
        self.main_interest = main_interest
        if not market.check_topic(main_interest):
            raise ValueError("Main interest is not in the market.")
        
        self.attention_bound = attention_bound
        self.index = index
        self._producer_following_rates = {i: 0 for i in range(market.num_producers)}
        self.delay_sensitivity = delay_sensitivity

    @property
    def producer_following_rates(self):
        return self._producer_following_rates
    
    @producer_following_rates.setter
    def producer_following_rates(self, value):
        if sum(value.values()) > self.attention_bound:
            raise ValueError("Sum of following rates exceeds attention bound.")
        self._producer_following_rates = value

    def get_following_rate_vector(self):
        return np.array([self.producer_following_rates[i] for i in range(self.market.num_producers)])
    
    def set_following_rate_vector(self, following_rate_vector: np.array):
        if len(following_rate_vector) != self.market.num_producers:
            raise ValueError("Following rate vector has wrong length.")
        self.producer_following_rates = {i: following_rate_vector[i] for i in range(self.market.num_producers)}

    def utility(self, topics: list[np.ndarray], production_rate) -> float:
        if len(topics) != self.market.num_producers:
            raise ValueError("Number of topics does not match number of producers.")
        
        reward = 0
        for consumer in self.market.consumers:
            for producer in self.market.producers:
                # TODO: check that consumer and producer aren't the same
                if not consumer.producer_following_rates[producer.index] > 0:
                    continue
                if not self.producer_following_rates[producer.index] > 0:
                    continue

                consumer_interest = producer.topic_probability(topics[producer.index]) * consumer.consumption_topic_interest(topics[producer.index])
                delay = np.exp(-self.delay_sensitivity * (1 / self.producer_following_rates[producer.index] + 1 / consumer.influencer_following_rates[self.index]))

                reward += production_rate * consumer_interest * delay

        return reward
