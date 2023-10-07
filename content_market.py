import numpy as np

from consumer import Consumer
from influencer import Influencer
from producer import Producer

class ContentMarket:
    """
    A content market where producers, consumers, and influencers react.
    The set of topics in the market is a rectangle in topics_bounds.shape[0] dimensions.
    """

    def __init__(self, topics_bounds: np.ndarray, num_producers, num_consumers, num_influencers):
        self.topics_bounds = topics_bounds
        self.topics_dim = topics_bounds.shape[0]
        self.producers: list[Producer] = []
        self.consumers: list[Consumer] = []
        self.influencers: list[Influencer] = []
        self.num_producers = num_producers
        self.num_consumers = num_consumers
        self.num_influencers = num_influencers

    def add_consumer(self, main_topic, topic_interest_function, attention_bound):
        if len(self.consumers) >= self.num_consumers:
            raise ValueError("Number of consumers exceeds limit.")
        self.consumers.append(Consumer(self, main_topic, topic_interest_function, attention_bound, len(self.consumers)))

    def add_producer(self, main_topic, topic_interest_function):
        if len(self.producers) >= self.num_producers:
            raise ValueError("Number of producers exceeds limit.")
        self.producers.append(Producer(self, main_topic, topic_interest_function, len(self.producers)))

    def add_influencer(self, main_topic, attention_bound):
        if len(self.influencers) >= self.num_influencers:
            raise ValueError("Number of influencers exceeds limit.")
        self.influencers.append(Influencer(self, main_topic, attention_bound, len(self.influencers)))

    def check_topic(self, topic: np.ndarray):
        if topic.shape != (self.topics_dim,):
            raise ValueError("Topic has wrong shape.")
        if not np.all(topic >= self.topics_bounds[:, 0]) or not np.all(topic <= self.topics_bounds[:, 1]):
            raise ValueError("Topic is not in the market.")
        
    