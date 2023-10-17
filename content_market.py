import numpy as np

from agents import Consumer, Producer, Influencer, Agent

from scipy.optimize import minimize, LinearConstraint

from util import OptimizationTargets

class ContentMarket:
    """
    A content market where producers, consumers, and influencers react.
    The set of topics in the market is a rectangle in topics_bounds.shape[0] dimensions.
    """

    def __init__(self, topics_bounds: np.ndarray):
        self.topics_bounds = topics_bounds
        self.topics_dim = topics_bounds.shape[0]
        self.agents: list[Agent] = []
        self.num_producers = 0
        self.num_consumers = 0
        self.num_influencers = 0
        self.num_agents = 0

    def add_agent(self, agent: Agent):
        self.agents.append(agent)
        self.num_agents += 1
        if isinstance(agent, Producer):
            self.num_producers += 1
        elif isinstance(agent, Consumer):
            self.num_consumers += 1
        elif isinstance(agent, Influencer):
            self.num_influencers += 1
        else:
            raise ValueError("Unknown agent type.")
        
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

    def finalize(self):
        for agent in self.agents:
            agent.init_following_rates()

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
                    attention_constraint = LinearConstraint(np.ones(self.num_agents + 1), lb=0, ub=consumer.attention_bound)
                    
                    bounds = []
                    for agent in self.agents:
                        if agent == consumer or (not isinstance(agent, Producer) and not isinstance(agent, Influencer)):
                            bounds.append((0, 0))
                        else:
                            bounds.append((0, None))
                    bounds.append((0, None)) # external

                    result = minimize(
                        fun=consumer.minimization_utility,
                        x0=consumer.get_following_rate_vector(),
                        args=(production_rate, external_production_rate, OptimizationTargets.CONSUMER),
                        constraints=attention_constraint,
                        bounds=bounds
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
                    attention_constraint = LinearConstraint(np.ones(self.num_agents + 1), lb=0, ub=influencer.attention_bound)

                    bounds = []
                    for agent in self.agents:
                        if agent == influencer or (not isinstance(agent, Producer) and not isinstance(agent, Influencer)):
                            bounds.append((0, 0))
                        else:
                            bounds.append((0, None))
                    bounds.append((influencer.get_following_rate_vector()[-1], influencer.get_following_rate_vector()[-1])) # external should not change when optimizing influencer

                    result = minimize(
                        fun=influencer.minimization_utility,
                        x0=influencer.get_following_rate_vector(),
                        args=(production_rate, external_production_rate, OptimizationTargets.INFLUENCER),
                        constraints=attention_constraint,
                        bounds=bounds
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
                        args=(production_rate, external_production_rate, OptimizationTargets.PRODUCER),
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