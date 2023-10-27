import numpy as np

from agents import Consumer, Producer, Influencer, Agent

from scipy.optimize import LinearConstraint

from util import OptimizationTargets, minimize_with_retry

from stats import ConsumerStats, ProducerStats, InfluencerStats, TestStats

from timeit import default_timer as timer

class ContentMarket:
    """
    A content market where producers, consumers, and influencers react.
    The set of topics in the market is a rectangle in topics_bounds.shape[0] dimensions.
    """

    def __init__(self, topics_bounds: np.ndarray, production_rate: float, external_production_rate: float):
        self.topics_bounds = topics_bounds
        self.topics_dim = topics_bounds.shape[0]
        self.agents: list[Agent] = []
        self.num_producers = 0
        self.num_consumers = 0
        self.num_influencers = 0
        self.num_agents = 0
        self.production_rate = production_rate
        self.external_production_rate = external_production_rate

    def add_agent(self, agent: Agent):
        self.agents.append(agent)
        self.num_agents += 1
        if isinstance(agent, Producer):
            self.num_producers += 1
        if isinstance(agent, Consumer):
            self.num_consumers += 1
        if isinstance(agent, Influencer):
            self.num_influencers += 1
        
        agent.set_market(self, len(self.agents) - 1)
        
    @property
    def producers(self) -> list[Producer]:
        return [agent for agent in self.agents if isinstance(agent, Producer)]
    
    @property
    def consumers(self) -> list[Consumer]:
        return [agent for agent in self.agents if isinstance(agent, Consumer)]
    
    @property
    def influencers(self) -> list[Influencer]:
        return [agent for agent in self.agents if isinstance(agent, Influencer)]
    
    def finalize(self, method="random"):
        """
        Finalize the market. This is done by setting the main interests of the producers and consumers.

        method: The method to use for setting the main interests. Can be "random" or "even".
        """
        if all(producer.main_interest is not None for producer in self.producers) and all(consumer.main_interest is not None for consumer in self.consumers):
            return
        if method == "random":
            self.rand_topics()
        elif method == "even":
            self.even_topics()
        else:
            raise ValueError("Unknown method.")
    
    def rand_topics(self):
        for agent in self.agents:
            if isinstance(agent, Producer) or isinstance(agent, Consumer):
                agent.set_main_interest(self.sample_topic())

    def even_topics(self):
        num_producers = self.num_producers
        num_consumers = self.num_consumers
        overlap = list(agent for agent in self.agents if isinstance(agent, Producer) and isinstance(agent, Consumer))
        num_overlap = len(overlap)

        topic_min = self.topics_bounds[:, 0]
        topic_max = self.topics_bounds[:, 1]

        producer_dist = list(tuple(topic) for topic in np.linspace(topic_min, topic_max, num_producers))
        consumer_dist = list(tuple(topic) for topic in np.linspace(topic_min, topic_max, num_consumers))

        # find the num_overlap pairs of values in producer_dist and consumer_dist that are closest to each other
        pairs = [(i, j) for i in producer_dist for j in consumer_dist]
        pairs.sort(key=lambda pair: np.linalg.norm(np.array(pair[0]) - np.array(pair[1])))

        used_producers = set()
        used_consumers = set()

        closest_pairs = []
        while len(closest_pairs) < num_overlap:
            pair = pairs.pop(0)
            if pair[0] in used_producers or pair[1] in used_consumers:
                continue
            closest_pairs.append(pair)
            used_producers.add(pair[0])
            used_consumers.add(pair[1])
        producer_dist = [producer for producer in producer_dist if producer not in used_producers]
        consumer_dist = [consumer for consumer in consumer_dist if consumer not in used_consumers]

        # now we set the pure producers and consumers to have interests from producer_dist and consumer_dist, respectively
        # but the overlapping ones should have interests that are the averages of elements in closest_pairs
        for producer in self.producers:
            if producer in overlap:
                continue
            producer.set_main_interest(np.array(producer_dist.pop()))
        for consumer in self.consumers:
            if consumer in overlap:
                continue
            consumer.set_main_interest(np.array(consumer_dist.pop()))
        for agent in overlap:
            pair = closest_pairs.pop()
            topic = np.mean(pair, axis=0)
            agent.set_main_interest(np.array(topic))

    def reset(self):
        """
        Reset the market to its initial state.
        """
        for agent in self.agents:
            agent.reset()

    def check_topic(self, topic: np.ndarray):
        if topic.shape != (self.topics_dim,):
            return False
        if not np.all(self.topics_bounds[:, 0] - topic < 1e-6) or not np.all(topic - self.topics_bounds[:, 1] < 1e-6):
            return False
        return True
        
    def sample_topic(self):
        """
        Generate a random topic in the market.
        """
        return np.array([np.random.uniform(self.topics_bounds[i, 0], self.topics_bounds[i, 1]) for i in range(self.topics_dim)])

    def optimize(self, max_iterations=100, converge_tol=1e-3) -> TestStats:
        """
        Optimize the market. This is done by iteratively optimizing the utility functions of the producers, consumers, and influencers.
        """
        self.finalize()
        self.reset()

        start_time = timer()

        stats = TestStats(market=self)
        
        for i in range(max_iterations):
            iter_start_time = timer()

            consumer_updates = {}
            consumer_times = {}
            influencer_updates = {}
            influencer_times = {}
            producer_updates = {}
            producer_times = {}

            if self.num_consumers > 0:
                for consumer in self.consumers:
                    cons_start_time = timer()
                    attention_constraint = LinearConstraint(np.ones(self.num_agents + 1), lb=0, ub=consumer.attention_bound)

                    print(f"Optimizing consumer {consumer.index}")
                    result = minimize_with_retry(
                        fun=consumer.minimization_utility,
                        x0=consumer.get_following_rate_vector(),
                        args=(self.production_rate, self.external_production_rate, OptimizationTargets.CONSUMER),
                        constraints=attention_constraint,
                        bounds=consumer.get_following_rate_bounds()
                    )

                    if not result.success:
                        raise RuntimeError("Optimization failed", result)

                    #consumer.set_following_rate_vector(result.x)
                    consumer_updates[consumer.index] = result.x

                    cons_end_time = timer()
                    consumer_times[consumer.index] = cons_end_time - cons_start_time


            if self.num_influencers > 0:
                # optimize influencers
                for influencer in self.influencers:
                    inf_start_time = timer()
                    attention_constraint = LinearConstraint(np.ones(self.num_agents + 1), lb=0, ub=influencer.attention_bound)

                    print(f"Optimizing influencer {influencer.index}")
                    result = minimize_with_retry(
                        fun=influencer.minimization_utility,
                        x0=influencer.get_following_rate_vector(),
                        args=(self.production_rate, self.external_production_rate, OptimizationTargets.INFLUENCER),
                        constraints=attention_constraint,
                        bounds=influencer.get_following_rate_bounds()
                    )

                    if not result.success:
                        raise RuntimeError("Optimization failed", result)

                    #influencer.set_following_rate_vector(result.x)
                    influencer_updates[influencer.index] = result.x

                    inf_end_time = timer()
                    influencer_times[influencer.index] = inf_end_time - inf_start_time

            if self.num_producers > 0:
                # optimize producers
                for producer in self.producers:
                    prod_start_time = timer()

                    print(f"Optimizing producer {producer.index}")
                    result = minimize_with_retry(
                        fun=producer.minimization_utility,
                        x0=producer.topic_produced,
                        args=(self.production_rate, self.external_production_rate, OptimizationTargets.PRODUCER),
                        bounds=self.topics_bounds,
                        num_retry=2
                    )

                    if not result.success:
                        raise RuntimeError("Optimization failed", result)

                    #producer.topic_produced = result.x
                    producer_updates[producer.index] = result.x

                    prod_end_time = timer()
                    producer_times[producer.index] = prod_end_time - prod_start_time

                    print(f"Optimization succeeded (overall {prod_end_time - prod_start_time}s): nit={result.nit}, nfev={result.nfev}, njev={result.njev}.")


            for consumer in self.consumers:
                consumer.set_following_rate_vector(consumer_updates[consumer.index])
            for influencer in self.influencers:
                influencer.set_following_rate_vector(influencer_updates[influencer.index])
            for producer in self.producers:
                producer.topic_produced = producer_updates[producer.index]

            iter_end_time = timer()
            
            stats.update(iter_end_time - iter_start_time, consumer_times, producer_times, influencer_times)

            print(f"Iteration {i} / {max_iterations} done in {iter_end_time - iter_start_time} seconds.")
            print(f"Total Social Welfare: {stats.total_social_welfare[-1]}")

            # check for convergence
            # we've converged if the following rates and topics don't change anymore
            # or if the utility doesn't change anymore
            if i > 0:
                if self.num_consumers > 0:
                    consumer_avg_rate_change = stats.average_consumer_rate_change[-1]
                    consumer_avg_utility_change_percent = stats.average_consumer_utility_change[-1] / (stats.average_consumer_utility[-1] + 1e-6)
                    print(f"Consumer rate change: {consumer_avg_rate_change}")
                    print(f"Consumer utility change: {consumer_avg_utility_change_percent}%")
                    consumer_convergence = consumer_avg_rate_change < converge_tol or consumer_avg_utility_change_percent < converge_tol
                else:
                    consumer_convergence = True

                if self.num_influencers > 0:
                    influencer_avg_rate_change = stats.average_influencer_rate_change[-1]
                    influencer_avg_utility_change_percent = stats.average_influencer_utility_change[-1] / (stats.average_influencer_utility[-1] + 1e-6)
                    influencer_convergence = influencer_avg_rate_change < converge_tol or influencer_avg_utility_change_percent < converge_tol
                    print(f"Influencer rate change: {influencer_avg_rate_change}")
                    print(f"Influencer utility change: {influencer_avg_utility_change_percent}%")
                else:
                    influencer_convergence = True

                if self.num_producers > 0:
                    producer_avg_topic_change = stats.average_producer_topic_change[-1]
                    producer_avg_utility_change_percent = stats.average_producer_utility_change[-1] / (stats.average_producer_utility[-1] + 1e-6)
                    producer_convergence = producer_avg_topic_change < converge_tol or producer_avg_utility_change_percent < converge_tol
                    print(f"Producer topic change: {producer_avg_topic_change}")
                    print(f"Producer utility change: {producer_avg_utility_change_percent}%")
                else:
                    producer_convergence = True
                if consumer_convergence and influencer_convergence and producer_convergence:
                    end_time = timer()
                    stats.finish(end_time - start_time)
                    print(f"Converged. Optimization took {end_time - start_time} seconds.")
                    break
                
        return stats