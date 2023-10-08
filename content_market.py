import numpy as np

from consumer import Consumer
from influencer import Influencer
from producer import Producer

from scipy.optimize import minimize, LinearConstraint

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
        
    def optimize(self, production_rate, external_production_rate, max_iterations=100):
        """
        Optimize the market. This is done by iteratively optimizing the utility functions of the producers, consumers, and influencers.
        """

        producer_topics = [producer.sample_topic() for producer in self.producers]
        
        for i in range(max_iterations):
            # optimize consumers
            for consumer in self.consumers:
                attention_constraint = LinearConstraint(np.ones(self.num_producers + self.num_influencers + 1), lb=0, ub=consumer.attention_bound)

                result = minimize(
                    fun=Consumer.minimization_utility,
                    x0=consumer.get_following_rate_vector(),
                    args=(consumer, producer_topics, production_rate, external_production_rate),
                    constraints=attention_constraint,
                )

                if not result.success:
                    raise RuntimeError("Optimization failed: " + result.message)

                consumer.set_following_rate_vector(result.x)

            # optimize influencers
            for influencer in self.influencers:
                attention_constraint = LinearConstraint(np.ones(self.num_producers), lb=0, ub=influencer.attention_bound)

                result = minimize(
                    fun=Influencer.minimization_utility,
                    x0=influencer.get_following_rate_vector(),
                    args=(influencer, production_rate, producer_topics),
                    constraints=attention_constraint,
                )

                if not result.success:
                    raise RuntimeError("Optimization failed: " + result.message)

                influencer.set_following_rate_vector(result.x)

            # optimize producers
            for producer in self.producers:
                result = minimize(
                    fun=Producer.minimization_utility,
                    x0=producer_topics[producer.index],
                    args=(producer, production_rate),
                )

                if not result.success:
                    raise RuntimeError("Optimization failed: " + result.message)

                producer.main_interest = result.x

            producer_topics = [producer.sample_topic() for producer in self.producers]

            print(f"Iteration {i} / {max_iterations} done.")

            # TODO: check if we are done