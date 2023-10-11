from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    from content_market import ContentMarket

import numpy as np

class Producer:

    def __init__(self, index: int, main_interest: np.ndarray, topic_interest_function):
        self.market = None
        self.main_interest = main_interest
        
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
    
    def sample_topic(self) -> np.ndarray:
        topic = self.main_interest + np.random.normal(0, 1, self.market.topics_dim)

        # clamp topic to be in the market
        lower_bounds = [bound[0] for bound in self.market.topics_bounds]
        upper_bounds = [bound[1] for bound in self.market.topics_bounds]
        topic = np.maximum(topic, lower_bounds)
        topic = np.minimum(topic, upper_bounds)

        return topic

    
    @staticmethod
    def utility(topic: np.ndarray, *args) -> float:
        producer = cast(Producer, args[0])
        if producer.market is None:
            raise ValueError("Producer has no market.")
        production_rate = cast(float, args[1])
        
        influencer_reward = 0
        for influencer in producer.market.influencers:
            for consumer in producer.market.consumers:
                if not influencer.producer_following_rates[producer.index] > 0:
                    continue
                if not consumer.influencer_following_rates[influencer.index] > 0:
                    continue
                if consumer == producer:
                    continue

                consumer_interest = producer.topic_probability(topic) * consumer.consumption_topic_interest(topic)
                delay = np.exp(-influencer.delay_sensitivity * (1 / influencer.producer_following_rates[producer.index] + 1 / consumer.influencer_following_rates[influencer.index]))

                influencer_reward += production_rate * consumer_interest * delay

        direct_consumer_reward = 0
        for consumer in producer.market.consumers:
            if not consumer.producer_following_rates[producer.index] > 0:
                continue
            if consumer == producer:
                continue

            consumer_interest = producer.topic_probability(topic) * consumer.consumption_topic_interest(topic)
            delay = np.exp(-consumer.delay_sensitivity * (1 / consumer.producer_following_rates[producer.index]))

            direct_consumer_reward += production_rate * consumer_interest * delay

        return influencer_reward + direct_consumer_reward
    
    @staticmethod
    def minimization_utility(topic: np.ndarray, *args) -> float:
        return -1 * Producer.utility(topic, *args)