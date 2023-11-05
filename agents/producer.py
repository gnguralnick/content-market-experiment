from typing import cast, TYPE_CHECKING
from agents.agent import Agent
if TYPE_CHECKING:
    from content_market import ContentMarket

import numpy as np

class Producer(Agent):

    def __init__(self, topic_interest_function):
        Agent.__init__(self)

        self.main_interest = None
        self.topic_produced = None
        self._production_topic_interest_function = topic_interest_function

    def set_main_interest(self, main_interest: np.ndarray):
        self.main_interest = main_interest

    def reset(self, position = 'main'):
        """
        Reset the producer. This will reset the topic produced according to the given method.
        main: the producer will produce content about its main interest
        random: the producer will produce content about a random topic
        center: the producer will produce content about the center of the market
        opposite: the producer will produce content about the opposite of its main interest
        farthest: the producer will produce content about the farthest topic from its main interest
        """
        if position == 'main' or position is None:
            self.topic_produced = self.main_interest
        elif position == 'random':
            self.topic_produced = self.market.sample_topic()
        elif position == 'center':
            topic_center = np.mean(self.market.topics_bounds, axis=1)
            self.topic_produced = topic_center
        elif position == 'opposite':
            topic_center = np.mean(self.market.topics_bounds, axis=1)
            self.topic_produced = topic_center + (topic_center - self.main_interest)
        elif position == 'farthest':
            topic_center = np.mean(self.market.topics_bounds, axis=1)
            topic = np.zeros(self.market.topics_dim)
            for i in range(self.market.topics_dim):
                if self.main_interest[i] < topic_center[i]:
                    topic[i] = self.market.topics_bounds[i][1]
                else:
                    topic[i] = self.market.topics_bounds[i][0]
            self.topic_produced = topic
        else:
            raise ValueError("Invalid position for producer reset.")

    def topic_probability(self, topic: np.ndarray) -> float:
        if not self.market.check_topic(topic):
            raise ValueError("Topic is not in the market.")
        distance = np.linalg.norm(topic - self.main_interest)
        return self._production_topic_interest_function(distance)

    def utility(self, x: np.ndarray, *args) -> float:
        if self.market is None or self.index is None:
            raise ValueError("Producer has no market.")
        production_rate = cast(float, args[0])

        topic = x
        
        influencer_reward = 0
        for influencer in self.market.influencers:
            if not influencer.following_rates[self.index] > 0:
                continue
            if influencer == self:
                continue
            for consumer in self.market.consumers:
                if not consumer.following_rates[influencer.index] > 0:
                    continue
                if consumer == self:
                    continue

                consumer_interest = self.topic_probability(topic) * consumer.consumption_topic_interest(topic)
                delay = np.exp(-influencer.delay_sensitivity * (1 / influencer.following_rates[self.index] + 1 / consumer.following_rates[influencer.index]))

                influencer_reward += production_rate * consumer_interest * delay

        direct_consumer_reward = 0
        for consumer in self.market.consumers:
            if not consumer.following_rates[self.index] > 0:
                continue
            if consumer == self:
                continue

            consumer_interest = self.topic_probability(topic) * consumer.consumption_topic_interest(topic)
            delay = np.exp(-consumer.delay_sensitivity * (1 / consumer.following_rates[self.index]))

            direct_consumer_reward += production_rate * consumer_interest * delay

        return influencer_reward + direct_consumer_reward