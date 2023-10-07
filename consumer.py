from content_market import ContentMarket
import numpy as np
from typing import cast


class Consumer:
    def __init__(self, market: ContentMarket, main_interest: np.ndarray, topic_interest_function, attention_bound, index, external_interest_prob, delay_sensitivity):
        self.market = market
        self.main_interest = main_interest
        if not market.check_topic(main_interest):
            raise ValueError("Main interest is not in the market.")

        self._topic_interest_function = topic_interest_function
        self.attention_bound = attention_bound
        self._producer_following_rates = {i: 0 for i in range(market.num_producers) if i != index}
        self._influencer_following_rates = {i: 0 for i in range(market.num_influencers)}
        self.index = index
        self._external_following_rate = 0
        self.external_interest_prob = external_interest_prob
        self.delay_sensitivity = delay_sensitivity


    def consumption_topic_interest(self, topic: np.ndarray) -> float:
        if not self.market.check_topic(topic):
            raise ValueError("Topic is not in the market.")
        distance = np.linalg.norm(topic - self.main_interest)
        return self._topic_interest_function(distance)

    @property
    def producer_following_rates(self):
        return self._producer_following_rates
    
    @producer_following_rates.setter
    def producer_following_rates(self, value):
        if sum(value.values()) + sum(self._influencer_following_rates.values()) + self._external_following_rate > self.attention_bound:
            raise ValueError("Sum of following rates exceeds attention bound.")
        self._producer_following_rates = value

    @property
    def influencer_following_rates(self):
        return self._influencer_following_rates
    
    @influencer_following_rates.setter
    def influencer_following_rates(self, value):
        if sum(value.values()) + sum(self._producer_following_rates.values()) + self._external_following_rate > self.attention_bound:
            raise ValueError("Sum of following rates exceeds attention bound.")
        self._influencer_following_rates = value

    @property
    def external_following_rate(self):
        return self._external_following_rate
    
    @external_following_rate.setter
    def external_following_rate(self, value):
        if sum(value.values()) + sum(self._producer_following_rates.values()) + sum(self._influencer_following_rates.values()) > self.attention_bound:
            raise ValueError("Sum of following rates exceeds attention bound.")
        self._external_following_rate = value

    def get_following_rate_vector(self):
        return np.array(list(self._producer_following_rates.values()) + list(self._influencer_following_rates.values()) + [self._external_following_rate])

    def set_following_rate_vector(self, vector: np.array):
        if len(vector) != self.market.num_producers + self.market.num_influencers + 1:
            raise ValueError("Vector has wrong length.")
        if sum(vector) > self.attention_bound:
            raise ValueError("Sum of following rates exceeds attention bound.")
        self._producer_following_rates = {i: vector[i] for i in range(self.market.num_producers)}
        self._influencer_following_rates = {i: vector[i + self.market.num_producers] for i in range(self.market.num_influencers)}
        self._external_following_rate = vector[-1]

    @staticmethod
    def utility(following_rate_vector: np.array, *args) -> float:
        consumer: Consumer = cast(Consumer, args[0])
        topics: list[np.ndarray] = cast(list[np.ndarray], args[1])
        production_rate: float = cast(float, args[2])
        external_production_rate: float = cast(float, args[3])

        if len(topics) != consumer.market.num_producers:
            raise ValueError("Number of topics does not match number of producers.")
        
        consumer.set_following_rate_vector(following_rate_vector)
        
        influencer_reward = 0
        for influencer in consumer.market.influencers:
            for producer in consumer.market.producers:
                if not influencer.producer_following_rates[producer.index] > 0:
                    continue
                if not consumer.influencer_following_rates[influencer.index] > 0:
                    continue

                # TODO: check that producer is not same as consumer

                topic_reward = producer.topic_probability(topics[producer.index]) * consumer.consumption_topic_interest(topics[producer.index])
                delay = np.exp(-consumer.delay_sensitivity * (1 / influencer.producer_following_rates[producer.index] + 1 / consumer.influencer_following_rates[influencer.index]))

                influencer_reward += production_rate * topic_reward * delay
        
        direct_following_reward = 0
        for producer in consumer.market.producers:
            if not consumer.producer_following_rates[producer.index] > 0:
                continue

            # TODO: check that producer is not same as consumer
            
            topic_reward = producer.topic_probability(topics[producer.index]) * consumer.consumption_topic_interest(topics[producer.index])
            delay = np.exp(-consumer.delay_sensitivity * (1 / consumer.producer_following_rates[producer.index]))
            direct_following_reward += production_rate * topic_reward * delay

        external_reward = 0
        if consumer.external_following_rate > 0:
            external_reward = external_production_rate * consumer.external_interest_prob * np.exp(-consumer.delay_sensitivity * (1 / consumer.external_following_rate))

        return influencer_reward + direct_following_reward + external_reward
