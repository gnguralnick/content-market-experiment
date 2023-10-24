import numpy as np
from agents import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from content_market import ContentMarket

class AgentStats:

    def __init__(self, agent: Agent, market: 'ContentMarket'):
        self.utilities = [0]
        self.utility_change = [0]
        self.market = market
        self.agent = agent

class ConsumerStats(AgentStats):

    def __init__(self, consumer: Consumer, market: 'ContentMarket'):
        super().__init__(consumer, market)
        self.consumer = consumer
        self.following_rates = [consumer.get_following_rate_vector()]
        self.attention_used = [sum(consumer.get_following_rate_vector())]
        self.rate_change = [0]

    def update(self):
        self.following_rates.append(self.consumer.get_following_rate_vector())
        self.attention_used.append(sum(self.consumer.get_following_rate_vector()))
        self.rate_change.append(np.linalg.norm(self.following_rates[-1] - self.following_rates[-2]))

        utility = self.consumer.utility(self.consumer.get_following_rate_vector(), self.market.production_rate, self.market.external_production_rate, OptimizationTargets.CONSUMER)
        self.utilities.append(utility)

        self.utility_change.append(self.utilities[-1] - self.utilities[-2])

class ProducerStats(AgentStats):

    def __init__(self, producer: Producer, market: 'ContentMarket'):
        super().__init__(producer, market)
        self.producer = producer
        self.topics = [producer.topic_produced]
        self.topic_change = [0]
        self.topic_distance = [np.linalg.norm(producer.topic_produced - producer.main_interest)]

    def update(self):
        self.topics.append(self.producer.topic_produced)
        self.topic_change.append(np.linalg.norm(self.topics[-1] - self.topics[-2]))
        self.topic_distance.append(np.linalg.norm(self.producer.topic_produced - self.producer.main_interest))

        utility = self.producer.utility(self.producer.topic_produced, self.market.production_rate, self.market.external_production_rate, OptimizationTargets.PRODUCER)
        self.utilities.append(utility)

        self.utility_change.append(self.utilities[-1] - self.utilities[-2])

class InfluencerStats(AgentStats):

    def __init__(self, influencer: Influencer, market: 'ContentMarket'):
        super().__init__(influencer, market)
        self.influencer = influencer
        self.following_rates = [influencer.get_following_rate_vector()]
        self.attention_used = [sum(influencer.get_following_rate_vector())]
        self.rate_change = [0]

    def update(self):
        self.following_rates.append(self.influencer.get_following_rate_vector())
        self.attention_used.append(sum(self.influencer.get_following_rate_vector()))
        self.rate_change.append(np.linalg.norm(self.following_rates[-1] - self.following_rates[-2]))

        utility = self.influencer.utility(self.influencer.get_following_rate_vector(), self.market.production_rate, self.market.external_production_rate, OptimizationTargets.INFLUENCER)
        self.utilities.append(utility)

        self.utility_change.append(self.utilities[-1] - self.utilities[-2])

class TestStats:

    def __init__(self, market: 'ContentMarket'):
        self.market = market
        self.num_iterations = 0
        self.total_consumer_utility = [0]
        self.total_producer_utility = [0]
        self.total_influencer_utility = [0]
        self.total_social_welfare = [0]
        self.average_consumer_rate_change = [0]
        self.average_producer_topic_change = [0]
        self.average_influencer_rate_change = [0]
        self.average_consumer_utility_change = [0]
        self.average_producer_utility_change = [0]
        self.average_influencer_utility_change = [0]
        self.consumer_stats = {consumer.index: ConsumerStats(consumer, self.market) for consumer in market.consumers}
        self.producer_stats = {producer.index: ProducerStats(producer, self.market) for producer in market.producers}
        self.influencer_stats = {influencer.index: InfluencerStats(influencer, self.market) for influencer in market.influencers}

    @property
    def average_consumer_utility_by_iteration(self):
        return [utility / self.market.num_consumers for utility in self.total_consumer_utility]
    
    @property
    def average_producer_utility_by_iteration(self):
        return [utility / self.market.num_producers for utility in self.total_producer_utility]
    
    @property
    def average_influencer_utility_by_iteration(self):
        return [utility / self.market.num_influencers for utility in self.total_influencer_utility]

    def update(self):
        self.num_iterations += 1
        for consumer in self.market.consumers:
            self.consumer_stats[consumer.index].update()
        for producer in self.market.producers:
            self.producer_stats[producer.index].update()
        for influencer in self.market.influencers:
            self.influencer_stats[influencer.index].update()

        self.total_consumer_utility.append(sum([self.consumer_stats[consumer.index].utilities[-1] for consumer in self.market.consumers]))
        self.total_producer_utility.append(sum([self.producer_stats[producer.index].utilities[-1] for producer in self.market.producers]))
        self.total_influencer_utility.append(sum([self.influencer_stats[influencer.index].utilities[-1] for influencer in self.market.influencers]))
        self.total_social_welfare.append(self.total_consumer_utility[-1] + self.total_producer_utility[-1] + self.total_influencer_utility[-1])

        self.average_consumer_rate_change.append(np.mean([self.consumer_stats[consumer.index].rate_change[-1] for consumer in self.market.consumers]))
        self.average_producer_topic_change.append(np.mean([self.producer_stats[producer.index].topic_change[-1] for producer in self.market.producers]))
        self.average_influencer_rate_change.append(np.mean([self.influencer_stats[influencer.index].rate_change[-1] for influencer in self.market.influencers]))

        self.average_consumer_utility_change.append(np.mean([self.consumer_stats[consumer.index].utility_change[-1] for consumer in self.market.consumers]))
        self.average_producer_utility_change.append(np.mean([self.producer_stats[producer.index].utility_change[-1] for producer in self.market.producers]))
        self.average_influencer_utility_change.append(np.mean([self.influencer_stats[influencer.index].utility_change[-1] for influencer in self.market.influencers]))
        