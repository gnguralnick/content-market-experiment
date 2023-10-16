import numpy as np

from agents import Consumer, Producer, Influencer

from scipy.optimize import minimize, LinearConstraint

class ContentMarket:
    """
    A content market where producers, consumers, and influencers react.
    The set of topics in the market is a rectangle in topics_bounds.shape[0] dimensions.
    """

    def __init__(self, topics_bounds: np.ndarray, num_producers: int, num_consumers: int, num_influencers: int):
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

    def add_producer(self, producer: Producer):
        if len(self.producers) >= self.num_producers:
            raise ValueError("Number of producers exceeds limit.")
        self.producers.append(producer)

    def add_influencer(self, influencer: Influencer):
        if len(self.influencers) >= self.num_influencers:
            raise ValueError("Number of influencers exceeds limit.")
        self.influencers.append(influencer)

    def finalize(self):
        for consumer in self.consumers:
            consumer.set_market(self)
        for producer in self.producers:
            producer.set_market(self)
        for influencer in self.influencers:
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
        self.finalize()

        consumer_stats = { 
            consumer.index: { 
                "following_rates": [consumer.get_following_rate_vector()], 
                "utilities": [0], 
                "rate_change": [0], 
                "attention_used": [sum(consumer.get_following_rate_vector())] 
            } for consumer in self.consumers 
        }
        influencer_stats = { 
            influencer.index: { 
                "following_rates": [influencer.get_following_rate_vector()], 
                "utilities": [0], 
                "rate_change": [0], 
                "attention_used": [sum(influencer.get_following_rate_vector())] 
            } for influencer in self.influencers 
        }
        producer_stats = { 
            producer.index: { 
                "topics": [producer.main_interest], 
                "utilities": [0], 
                "topic_change": [0] 
            } for producer in self.producers 
        }
        total_stats = { 
            "consumer_utilities": [0], 
            "influencer_utilities": [0], 
            "producer_utilities": [0], 
            "social_welfare": [0]
        }
        average_stats = { 
            "consumer_utilities": [0], 
            "influencer_utilities": [0], 
            "producer_utilities": [0], 
            "consumer_rate_change": [0], 
            "influencer_rate_change": [0], 
            "producer_topic_change": [0], 
            "consumer_attention_used": [sum(sum(consumer.get_following_rate_vector()) for consumer in self.consumers) / self.num_consumers] if self.num_consumers > 0 else [0],
            "influencer_attention_used": [sum(sum(influencer.get_following_rate_vector()) for influencer in self.influencers) / self.num_influencers] if self.num_influencers > 0 else [0],
        }
        
        for i in range(max_iterations):
            # optimize consumers
            total_stats["consumer_utilities"].append(0)
            total_stats["influencer_utilities"].append(0)
            total_stats["producer_utilities"].append(0)
            total_stats["social_welfare"].append(0)

            if self.num_consumers > 0:
                for consumer in self.consumers:
                    attention_constraint = LinearConstraint(np.ones(self.num_producers + self.num_influencers + 1), lb=0, ub=consumer.attention_bound)

                    result = minimize(
                        fun=consumer.minimization_utility,
                        x0=consumer.get_following_rate_vector(),
                        args=(production_rate, external_production_rate),
                        constraints=attention_constraint,
                        bounds=[(0, None) for _ in range(self.num_producers + self.num_influencers)] + [(0, consumer.attention_bound)]
                    )

                    if not result.success:
                        raise RuntimeError("Optimization failed", result)

                    consumer.set_following_rate_vector(result.x)

                    rate_change = np.linalg.norm(result.x - consumer_stats[consumer.index]["following_rates"][-1])

                    consumer_stats[consumer.index]["following_rates"].append(result.x)
                    consumer_stats[consumer.index]["rate_change"].append(rate_change)
                    consumer_stats[consumer.index]["utilities"].append(-result.fun)
                    consumer_stats[consumer.index]["attention_used"].append(sum(result.x))
                    total_stats["consumer_utilities"][-1] += -result.fun
                    total_stats["social_welfare"][-1] += -result.fun
                average_stats["consumer_utilities"].append(total_stats["consumer_utilities"][-1] / self.num_consumers)
                average_stats["consumer_rate_change"].append(np.mean([consumer_stats[consumer.index]["rate_change"][-1] for consumer in self.consumers]))
                average_stats["consumer_attention_used"].append(np.mean([consumer_stats[consumer.index]["attention_used"][-1] for consumer in self.consumers]))


            if self.num_influencers > 0:
                # optimize influencers
                for influencer in self.influencers:
                    attention_constraint = LinearConstraint(np.ones(self.num_producers), lb=0, ub=influencer.attention_bound)

                    result = minimize(
                        fun=influencer.minimization_utility,
                        x0=influencer.get_following_rate_vector(),
                        args=(production_rate),
                        constraints=attention_constraint,
                        bounds=[(0, None) for _ in range(self.num_producers)]
                    )

                    if not result.success:
                        raise RuntimeError("Optimization failed", result)

                    influencer.set_following_rate_vector(result.x)

                    rate_change = np.linalg.norm(result.x - influencer_stats[influencer.index]["following_rates"][-1])

                    influencer_stats[influencer.index]["following_rates"].append(result.x)
                    influencer_stats[influencer.index]["rate_change"].append(rate_change)
                    influencer_stats[influencer.index]["utilities"].append(-result.fun)
                    influencer_stats[influencer.index]["attention_used"].append(sum(result.x))
                    total_stats["influencer_utilities"][-1] += -result.fun
                    total_stats["social_welfare"][-1] += -result.fun
                average_stats["influencer_utilities"].append(total_stats["influencer_utilities"][-1] / self.num_influencers)
                average_stats["influencer_rate_change"].append(np.mean([influencer_stats[influencer.index]["rate_change"][-1] for influencer in self.influencers]))
                average_stats["influencer_attention_used"].append(np.mean([influencer_stats[influencer.index]["attention_used"][-1] for influencer in self.influencers]))

            if self.num_producers > 0:
                # optimize producers
                for producer in self.producers:

                    result = minimize(
                        fun=producer.minimization_utility,
                        x0=producer.topic_produced,
                        args=(production_rate),
                        bounds=self.topics_bounds,
                    )

                    if not result.success:
                        raise RuntimeError("Optimization failed", result)

                    producer.topic_produced = result.x

                    topic_change = np.linalg.norm(result.x - producer_stats[producer.index]["topics"][-1])

                    producer_stats[producer.index]["topics"].append(result.x)
                    producer_stats[producer.index]["topic_change"].append(topic_change)
                    producer_stats[producer.index]["utilities"].append(-result.fun)
                    total_stats["producer_utilities"][-1] += -result.fun
                    total_stats["social_welfare"][-1] += -result.fun
                average_stats["producer_utilities"].append(total_stats["producer_utilities"][-1] / self.num_producers)
                average_stats["producer_topic_change"].append(np.mean([producer_stats[producer.index]["topic_change"][-1] for producer in self.producers]))

            print(f"Iteration {i} / {max_iterations} done.")
            print(f"Total Social Welfare: {total_stats['social_welfare'][-1]}")

            # check for convergence
            # we've converged if the following rates and topics don't change anymore
            # or if the utility doesn't change anymore
            if i > 0:
                if self.num_consumers > 0:
                    consumer_avg_rate_change = average_stats["consumer_rate_change"][-1]
                    consumer_avg_utility_change = abs(average_stats["consumer_utilities"][-1] - average_stats["consumer_utilities"][-2])
                    print(f"Consumer rate change: {consumer_avg_rate_change}")
                    print(f"Consumer utility change: {consumer_avg_utility_change}")
                    consumer_convergence = consumer_avg_rate_change < 1e-6 or consumer_avg_utility_change < 1e-6
                else:
                    consumer_convergence = True

                if self.num_influencers > 0:
                    influencer_avg_rate_change = average_stats["influencer_rate_change"][-1]
                    influencer_avg_utility_change = abs(average_stats["influencer_utilities"][-1] - average_stats["influencer_utilities"][-2])
                    influencer_convergence = influencer_avg_rate_change < 1e-6 or influencer_avg_utility_change < 1e-6
                    print(f"Influencer rate change: {influencer_avg_rate_change}")
                    print(f"Influencer utility change: {influencer_avg_utility_change}")
                else:
                    influencer_convergence = True

                if self.num_producers > 0:
                    producer_avg_topic_change = average_stats["producer_topic_change"][-1]
                    producer_avg_utility_change = abs(average_stats["producer_utilities"][-1] - average_stats["producer_utilities"][-2])
                    producer_convergence = producer_avg_topic_change < 1e-6 or producer_avg_utility_change < 1e-6
                    print(f"Producer topic change: {producer_avg_topic_change}")
                    print(f"Producer utility change: {producer_avg_utility_change}")
                else:
                    producer_convergence = True
                if consumer_convergence and influencer_convergence and producer_convergence:
                    print("Converged.")
                    break
                
        
        return consumer_stats, influencer_stats, producer_stats, total_stats, average_stats