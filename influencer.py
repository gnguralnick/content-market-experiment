import numpy as np
from content_market import ContentMarket

class Influencer:

    def __init__(self, market: ContentMarket, main_interest: np.ndarray, attention_bound, index):
        self.market = market
        self.main_interest = main_interest
        if not market.check_topic(main_interest):
            raise ValueError("Main interest is not in the market.")
        
        self.attention_bound = attention_bound
        self.index = index
        self._producer_following_rates = {i: 0 for i in range(market.num_producers)}

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