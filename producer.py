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
        return np.array([np.random.uniform(self.market.topics_bounds[i, 0], self.market.topics_bounds[i, 1]) for i in range(self.market.topics_dim)])
    
    def utility(self, topics: list[np.ndarray], production_rate) -> float:
        if len(topics) != self.market.num_producers:
            raise ValueError("Number of topics does not match number of producers.")
        
        influencer_reward = 0
        for influencer in self.market.influencers:
            for consumer in self.market.consumers:
                if not influencer.producer_following_rates[self.index] > 0:
                    continue
                if not consumer.influencer_following_rates[influencer.index] > 0:
                    continue
                # TODO: check that consumer and producer (self) aren't the same

                consumer_interest = self.topic_probability(topics[self.index]) * consumer.consumption_topic_interest(topics[self.index])
                delay = np.exp(-influencer.delay_sensitivity * (1 / influencer.producer_following_rates[self.index] + 1 / consumer.influencer_following_rates[influencer.index]))

                influencer_reward += production_rate * consumer_interest * delay

        direct_consumer_reward = 0
        for consumer in self.market.consumers:
            if not consumer.producer_following_rates[self.index] > 0:
                continue
            # TODO: check that consumer and producer (self) aren't the same

            consumer_interest = self.topic_probability(topics[self.index]) * consumer.consumption_topic_interest(topics[self.index])
            delay = np.exp(-consumer.delay_sensitivity * (1 / consumer.producer_following_rates[self.index]))

            direct_consumer_reward += production_rate * consumer_interest * delay

        return influencer_reward + direct_consumer_reward