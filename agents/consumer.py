import numpy as np
from typing import cast, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from content_market import ContentMarket


class Consumer:
    def __init__(self, index: int, main_interest: np.ndarray, topic_interest_function: Callable[[float], float], attention_bound, external_interest_prob, delay_sensitivity):
        self.market = None
        self.main_interest = main_interest

        self._producer_following_rates = dict()
        self._influencer_following_rates = dict()

        self._topic_interest_function = topic_interest_function
        self.attention_bound = attention_bound
        
        self.index = index
        self._external_following_rate = 0
        self.external_interest_prob = external_interest_prob
        self.delay_sensitivity = delay_sensitivity

    def set_market(self, market: 'ContentMarket'):
        self.market = market
        if not market.check_topic(self.main_interest):
            raise ValueError("Main interest is not in the market.")
        
        # self._producer_following_rates = {i: 0 for i in range(market.num_producers)}
        # self._influencer_following_rates = {i: 0 for i in range(market.num_influencers)}

        cur_sum = 0
        for i in range(market.num_producers):
            self._producer_following_rates[i] = np.random.uniform(0, self.attention_bound - cur_sum) / 2
            cur_sum += self._producer_following_rates[i]

        for i in range(market.num_influencers):
            self._influencer_following_rates[i] = np.random.uniform(0, self.attention_bound - cur_sum) / 2
            cur_sum += self._influencer_following_rates[i]
        
        self._external_following_rate = np.random.uniform(0, self.attention_bound - cur_sum)

        # num_follows = self.market.num_producers + self.market.num_influencers + 1
        # rate_per_follow = self.attention_bound / num_follows

        # self._producer_following_rates = {i: rate_per_follow for i in range(market.num_producers)}
        # self._influencer_following_rates = {i: rate_per_follow for i in range(market.num_influencers)}
        # self._external_following_rate = rate_per_follow

    def consumption_topic_interest(self, topic: np.ndarray) -> float:
        if not self.market.check_topic(topic):
            raise ValueError("Topic is not in the market.")
        distance = np.linalg.norm(topic - self.main_interest)
        return self._topic_interest_function(distance)

    @property
    def producer_following_rates(self) -> dict[int, float]:
        return self._producer_following_rates
    
    @producer_following_rates.setter
    def producer_following_rates(self, value):
        if sum(value.values()) + sum(self._influencer_following_rates.values()) + self._external_following_rate - self.attention_bound > 1e-6:
            raise ValueError("Sum of following rates exceeds attention bound.")
        self._producer_following_rates = value

    @property
    def influencer_following_rates(self) -> dict[int, float]:
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
        if sum(vector) - self.attention_bound > 1e-6:
            raise ValueError("Sum of following rates exceeds attention bound.")
        self._producer_following_rates = {i: vector[i] for i in range(self.market.num_producers)}
        self._influencer_following_rates = {i: vector[i + self.market.num_producers] for i in range(self.market.num_influencers)}
        self._external_following_rate = vector[-1]

    def utility(self, following_rate_vector: np.array, *args) -> float:
        if self.market is None:
            raise ValueError("Consumer has no market.")
        production_rate: float = cast(float, args[0])
        external_production_rate: float = cast(float, args[1])
        
        self.set_following_rate_vector(following_rate_vector)
        
        influencer_reward = 0
        for influencer in self.market.influencers:
            for producer in self.market.producers:
                if not influencer.producer_following_rates[producer.index] > 0:
                    continue
                if not self.influencer_following_rates[influencer.index] > 0:
                    continue

                if self == producer:
                    continue

                topic_reward = producer.topic_probability(producer.topic_produced) * self.consumption_topic_interest(producer.topic_produced)
                delay = np.exp(-self.delay_sensitivity * (1 / influencer.producer_following_rates[producer.index] + 1 / self.influencer_following_rates[influencer.index]))

                influencer_reward += production_rate * topic_reward * delay
        
        direct_following_reward = 0
        for producer in self.market.producers:
            if not self.producer_following_rates[producer.index] > 0:
                continue

            if self == producer:
                continue

            topic_reward = producer.topic_probability(producer.topic_produced) * self.consumption_topic_interest(producer.topic_produced)
            delay = np.exp(-self.delay_sensitivity * (1 / self.producer_following_rates[producer.index]))
            direct_following_reward += production_rate * topic_reward * delay

        external_reward = 0
        if self.external_following_rate > 0:
            external_reward = external_production_rate * self.external_interest_prob * np.exp(-self.delay_sensitivity * (1 / self.external_following_rate))

        return influencer_reward + direct_following_reward + external_reward

    def minimization_utility(self, following_rate_vector: np.ndarray, *args) -> float:
        return -1 * self.utility(following_rate_vector, *args)