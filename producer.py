import numpy as np
from content_market import ContentMarket

class Producer:

    def __init__(self, market: ContentMarket, main_interest: np.ndarray, topic_interest_function, index):
        self.market = market
        self.main_interest = main_interest
        if not market.check_topic(main_interest):
            raise ValueError("Main interest is not in the market.")
        
        self._topic_interest_function = topic_interest_function
        self.index = index

    def topic_probability(self, topic: np.ndarray) -> float:
        if not self.market.check_topic(topic):
            raise ValueError("Topic is not in the market.")
        distance = np.linalg.norm(topic - self.main_interest)
        return self._topic_interest_function(distance)
    
    def sample_topic(self):
        # generate a random topic t in the market
        # such that the probability of t is proportional to self.topic_probability(t)

        raise NotImplementedError("Implement this method.")