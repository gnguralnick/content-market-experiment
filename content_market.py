import numpy as np

from agents import Consumer, Producer, Influencer

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

    def add_consumer(self, consumer: Consumer):
        if len(self.consumers) >= self.num_consumers:
            raise ValueError("Number of consumers exceeds limit.")
        self.consumers.append(consumer)
        consumer.set_market(self)

    def add_producer(self, producer: Producer):
        if len(self.producers) >= self.num_producers:
            raise ValueError("Number of producers exceeds limit.")
        self.producers.append(producer)
        producer.set_market(self)

    def add_influencer(self, influencer: Influencer):
        if len(self.influencers) >= self.num_influencers:
            raise ValueError("Number of influencers exceeds limit.")
        self.influencers.append(influencer)
        influencer.set_market(self)

    def check_topic(self, topic: np.ndarray):
        if topic.shape != (self.topics_dim,):
            return False
        if not np.all(topic >= self.topics_bounds[:, 0]) or not np.all(topic <= self.topics_bounds[:, 1]):
            return False
        return True
        
    def sample_topic(self):
        # generate a random topic t in the market
        return np.array([np.random.uniform(self.topics_bounds[i, 0], self.topics_bounds[i, 1]) for i in range(self.topics_dim)])
        
    def optimize(self, production_rate, external_production_rate, max_iterations=100):
        """
        Optimize the market. This is done by iteratively optimizing the utility functions of the producers, consumers, and influencers.
        """

        producer_topics = [self.sample_topic() for producer in self.producers]

        consumer_utilities = []
        influencer_utilities = []
        producer_utilities = []
        
        for i in range(max_iterations):
            # optimize consumers
            consumer_utilities.append([])
            influencer_utilities.append([])
            producer_utilities.append([])
            for consumer in self.consumers:
                attention_constraint = LinearConstraint(np.ones(self.num_producers + self.num_influencers + 1), lb=0, ub=consumer.attention_bound)

                result = minimize(
                    fun=Consumer.minimization_utility,
                    x0=consumer.get_following_rate_vector(),
                    args=(consumer, producer_topics, production_rate, external_production_rate),
                    constraints=attention_constraint,
                )

                if not result.success:
                    raise RuntimeError("Optimization failed", result)

                consumer.set_following_rate_vector(result.x)

                consumer_utilities[-1].append(-result.fun)

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
                    raise RuntimeError("Optimization failed", result)

                influencer.set_following_rate_vector(result.x)

                influencer_utilities[-1].append(-result.fun)

            # optimize producers
            for producer in self.producers:

                result = minimize(
                    fun=Producer.minimization_utility,
                    x0=producer_topics[producer.index],
                    args=(producer, production_rate),
                    bounds=self.topics_bounds,
                )

                if not result.success:
                    raise RuntimeError("Optimization failed", result)

                producer.main_interest = result.x

                producer_utilities[-1].append(-result.fun)

            producer_topics = [self.sample_topic() for producer in self.producers]

            print(f"Iteration {i} / {max_iterations} done.")
            print(f"\tConsumer utilities: {consumer_utilities[-1]}")
            print(f"\tInfluencer utilities: {influencer_utilities[-1]}")
            print(f"\tProducer utilities: {producer_utilities[-1]}")

            # TODO: check if we are done
        
        return consumer_utilities, influencer_utilities, producer_utilities