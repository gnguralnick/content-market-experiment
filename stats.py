import numpy as np
from agents import *
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from content_market import ContentMarket

class AgentStats:

    def __init__(self, agent: Agent, market: 'ContentMarket'):
        self.utilities = [0]
        self.utility_change = [0]
        self.market = market
        self.agent = agent
        self.optimization_times = []

    def to_dict(self):
        return {
            'utilities': self.utilities,
            'utility_change': self.utility_change,
            'optimization_times': self.optimization_times,
            'agent': self.agent.to_dict(),
            'index': self.agent.index,
        }

class ConsumerStats(AgentStats):

    def __init__(self, consumer: Consumer, market: 'ContentMarket'):
        super().__init__(consumer, market)
        self.consumer = consumer
        self.following_rates = [consumer.get_following_rate_vector()]
        self.attention_used = [sum(consumer.get_following_rate_vector())]
        self.rate_change = [0]

    def update(self, optimization_time):
        self.following_rates.append(self.consumer.get_following_rate_vector())
        self.attention_used.append(sum(self.consumer.get_following_rate_vector()))
        self.rate_change.append(np.linalg.norm(self.following_rates[-1] - self.following_rates[-2]))

        utility = self.consumer.utility(self.consumer.get_following_rate_vector(), self.market.production_rate, self.market.external_production_rate, OptimizationTargets.CONSUMER)
        self.utilities.append(utility)

        self.utility_change.append(self.utilities[-1] - self.utilities[-2])

        self.optimization_times.append(optimization_time)
    
    def get_follow_proportion(self, agent: str):
        if agent == 'external':
            index = -1
        elif agent == 'influencer':
            index = self.market.influencers[0].index
        else:
            index = int(agent)
        follow = []
        for vec in self.following_rates:
            follow.append(vec[index])
        return [follow[i] / self.attention_used[i] for i in range(len(follow))]
    
    @property
    def num_producers_followed(self):
        return [sum([1 if vec[i] > 0 else 0 for i in range(self.market.num_producers)]) for vec in self.following_rates]
    
    def to_dict(self):
        return super().to_dict() | {
            'following_rates': self.following_rates,
            'attention_used': self.attention_used,
            'rate_change': self.rate_change,
        }

    @staticmethod
    def from_dict(consumer_stats_dict: dict, market: 'ContentMarket'):
        consumer_dict = consumer_stats_dict['agent']
        consumer = Consumer.from_dict(consumer_dict, market)
        consumer_stats = ConsumerStats(consumer, market)
        consumer_stats.utilities = consumer_stats_dict['utilities']
        consumer_stats.utility_change = consumer_stats_dict['utility_change']
        consumer_stats.optimization_times = consumer_stats_dict['optimization_times']
        consumer_stats.following_rates = consumer_stats_dict['following_rates']
        consumer_stats.attention_used = consumer_stats_dict['attention_used']
        consumer_stats.rate_change = consumer_stats_dict['rate_change']
        return consumer_stats

class ProducerStats(AgentStats):

    def __init__(self, producer: Producer, market: 'ContentMarket'):
        super().__init__(producer, market)
        self.producer = producer
        self.topics = [producer.topic_produced]
        self.topic_change = [0]
        self.topic_distance = [np.linalg.norm(producer.topic_produced - producer.main_interest)]

    def update(self, optimization_time):
        self.topics.append(self.producer.topic_produced)
        self.topic_change.append(np.linalg.norm(self.topics[-1] - self.topics[-2]))
        self.topic_distance.append(np.linalg.norm(self.producer.topic_produced - self.producer.main_interest))

        utility = self.producer.utility(self.producer.topic_produced, self.market.production_rate, self.market.external_production_rate, OptimizationTargets.PRODUCER)
        self.utilities.append(utility)

        self.utility_change.append(self.utilities[-1] - self.utilities[-2])

        self.optimization_times.append(optimization_time)

    def to_dict(self):
        return super().to_dict() | {
            'topics': self.topics,
            'topic_change': self.topic_change,
            'topic_distance': self.topic_distance,
        }

    @staticmethod
    def from_dict(producer_stats_dict: dict, market: 'ContentMarket'):
        producer_dict = producer_stats_dict['agent']
        producer = Producer.from_dict(producer_dict, market)
        producer_stats = ProducerStats(producer, market)
        producer_stats.utilities = producer_stats_dict['utilities']
        producer_stats.utility_change = producer_stats_dict['utility_change']
        producer_stats.optimization_times = producer_stats_dict['optimization_times']
        producer_stats.topics = producer_stats_dict['topics']
        producer_stats.topic_change = producer_stats_dict['topic_change']
        producer_stats.topic_distance = producer_stats_dict['topic_distance']
        return producer_stats

class InfluencerStats(AgentStats):

    def __init__(self, influencer: Influencer, market: 'ContentMarket'):
        super().__init__(influencer, market)
        self.influencer = influencer
        self.following_rates = [influencer.get_following_rate_vector()]
        self.attention_used = [sum(influencer.get_following_rate_vector())]
        self.rate_change = [0]

    def update(self, optimization_time):
        self.following_rates.append(self.influencer.get_following_rate_vector())
        self.attention_used.append(sum(self.influencer.get_following_rate_vector()))
        self.rate_change.append(np.linalg.norm(self.following_rates[-1] - self.following_rates[-2]))

        utility = self.influencer.utility(self.influencer.get_following_rate_vector(), self.market.production_rate, self.market.external_production_rate, OptimizationTargets.INFLUENCER)
        self.utilities.append(utility)

        self.utility_change.append(self.utilities[-1] - self.utilities[-2])

        self.optimization_times.append(optimization_time)

    def to_dict(self):
        return super().to_dict() | {
            'following_rates': self.following_rates,
            'attention_used': self.attention_used,
            'rate_change': self.rate_change,
        }

    @staticmethod
    def from_dict(influencer_stats_dict: dict, market: 'ContentMarket'):
        influencer_dict = influencer_stats_dict['agent']
        influencer = Influencer.from_dict(influencer_dict, market)
        influencer_stats = InfluencerStats(influencer, market)
        influencer_stats.utilities = influencer_stats_dict['utilities']
        influencer_stats.utility_change = influencer_stats_dict['utility_change']
        influencer_stats.optimization_times = influencer_stats_dict['optimization_times']
        influencer_stats.following_rates = influencer_stats_dict['following_rates']
        influencer_stats.attention_used = influencer_stats_dict['attention_used']
        influencer_stats.rate_change = influencer_stats_dict['rate_change']
        return influencer_stats

class TestStats:

    def __init__(self, market: 'ContentMarket'):
        self.market = market
        self.finished = False
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

        self.optimization_time = 0
        self.optimization_times = []

    @property
    def average_consumer_utility(self):
        return [utility / self.market.num_consumers for utility in self.total_consumer_utility]
    
    @property
    def average_producer_utility(self):
        return [utility / self.market.num_producers for utility in self.total_producer_utility]
    
    @property
    def average_influencer_utility(self):
        return [utility / self.market.num_influencers for utility in self.total_influencer_utility]
    
    def get_average_follow_proportion_by_iteration(self, agent: str):
        return [np.mean([self.consumer_stats[consumer.index].get_follow_proportion(agent)[i] for consumer in self.market.consumers]) for i in range(self.num_iterations + 1)]
    
    @property
    def average_influencer_follow_proportion(self):
        return self.get_average_follow_proportion_by_iteration('influencer')
    
    @property
    def average_external_follow_proportion(self):
        return self.get_average_follow_proportion_by_iteration('external')
    
    @property
    def average_producer_topic_distance_from_main_interest(self):
        return [np.mean([self.producer_stats[producer.index].topic_distance[i] for producer in self.market.producers]) for i in range(self.num_iterations + 1)]
    
    @property
    def producer_topic_standard_deviation(self):
        return [np.std([self.producer_stats[producer.index].topics[i] for producer in self.market.producers]) for i in range(self.num_iterations + 1)]
    
    @property
    def average_consumer_num_producers_followed(self):
        return [np.mean([self.consumer_stats[consumer.index].num_producers_followed[i] for consumer in self.market.consumers]) for i in range(self.num_iterations + 1)]
    
    @property
    def average_following_rate_by_main_interest_closeness(self):
        following_rates_by_main_interest_closeness = defaultdict(list)
        for consumer in self.market.consumers:
            for producer in sorted(self.market.producers, key=lambda x: np.linalg.norm(x.main_interest - consumer.main_interest)):
                if producer == consumer:
                    continue
                interest_closeness = np.linalg.norm(producer.main_interest - consumer.main_interest)
                following_rates_by_main_interest_closeness[interest_closeness].append(self.consumer_stats[consumer.index].following_rates[-1][producer.index])

        return sorted([(interest_closeness, np.mean(following_rates)) for interest_closeness, following_rates in following_rates_by_main_interest_closeness.items()], key=lambda x: x[0])
            
    
    def finish(self, optimization_time):
        self.optimization_time = optimization_time
        self.finished = True

    def update(self, optimization_time, consumer_times, producer_times, influencer_times):
        if self.finished:
            raise ValueError("Cannot update finished test.")
        self.num_iterations += 1
        for consumer in self.market.consumers:
            self.consumer_stats[consumer.index].update(consumer_times[consumer.index])
        for producer in self.market.producers:
            self.producer_stats[producer.index].update(producer_times[producer.index])
        for influencer in self.market.influencers:
            self.influencer_stats[influencer.index].update(influencer_times[influencer.index])

        self.total_consumer_utility.append(sum([self.consumer_stats[consumer.index].utilities[-1] for consumer in self.market.consumers]))
        self.total_producer_utility.append(sum([self.producer_stats[producer.index].utilities[-1] for producer in self.market.producers]))
        self.total_influencer_utility.append(sum([self.influencer_stats[influencer.index].utilities[-1] for influencer in self.market.influencers]))
        self.total_social_welfare.append(self.total_consumer_utility[-1])

        self.average_consumer_rate_change.append(np.mean([self.consumer_stats[consumer.index].rate_change[-1] for consumer in self.market.consumers]))
        self.average_producer_topic_change.append(np.mean([self.producer_stats[producer.index].topic_change[-1] for producer in self.market.producers]))
        self.average_influencer_rate_change.append(np.mean([self.influencer_stats[influencer.index].rate_change[-1] for influencer in self.market.influencers]))

        self.average_consumer_utility_change.append(np.mean([self.consumer_stats[consumer.index].utility_change[-1] for consumer in self.market.consumers]))
        self.average_producer_utility_change.append(np.mean([self.producer_stats[producer.index].utility_change[-1] for producer in self.market.producers]))
        self.average_influencer_utility_change.append(np.mean([self.influencer_stats[influencer.index].utility_change[-1] for influencer in self.market.influencers]))

        self.optimization_times.append(optimization_time)

    def to_dict(self):
        return {
            'num_iterations': self.num_iterations,
            'total_consumer_utility': self.total_consumer_utility,
            'total_producer_utility': self.total_producer_utility,
            'total_influencer_utility': self.total_influencer_utility,
            'total_social_welfare': self.total_social_welfare,
            'average_consumer_rate_change': self.average_consumer_rate_change,
            'average_producer_topic_change': self.average_producer_topic_change,
            'average_influencer_rate_change': self.average_influencer_rate_change,
            'average_consumer_utility_change': self.average_consumer_utility_change,
            'average_producer_utility_change': self.average_producer_utility_change,
            'average_influencer_utility_change': self.average_influencer_utility_change,
            'consumer_stats': {consumer.index: self.consumer_stats[consumer.index].to_dict() for consumer in self.market.consumers},
            'producer_stats': {producer.index: self.producer_stats[producer.index].to_dict() for producer in self.market.producers},
            'influencer_stats': {influencer.index: self.influencer_stats[influencer.index].to_dict() for influencer in self.market.influencers},
            'optimization_time': self.optimization_time,
            'optimization_times': self.optimization_times,
            'market': self.market.to_dict(),
        }

    @staticmethod
    def from_dict(test_stats_dict: dict):
        market_dict = test_stats_dict['market']
        market = ContentMarket.from_dict(market_dict)
        test_stats = TestStats(market)
        test_stats.num_iterations = test_stats_dict['num_iterations']
        test_stats.total_consumer_utility = test_stats_dict['total_consumer_utility']
        test_stats.total_producer_utility = test_stats_dict['total_producer_utility']
        test_stats.total_influencer_utility = test_stats_dict['total_influencer_utility']
        test_stats.total_social_welfare = test_stats_dict['total_social_welfare']
        test_stats.average_consumer_rate_change = test_stats_dict['average_consumer_rate_change']
        test_stats.average_producer_topic_change = test_stats_dict['average_producer_topic_change']
        test_stats.average_influencer_rate_change = test_stats_dict['average_influencer_rate_change']
        test_stats.average_consumer_utility_change = test_stats_dict['average_consumer_utility_change']
        test_stats.average_producer_utility_change = test_stats_dict['average_producer_utility_change']
        test_stats.average_influencer_utility_change = test_stats_dict['average_influencer_utility_change']
        test_stats.consumer_stats = {index: ConsumerStats.from_dict(consumer_stats_dict, market) for index, consumer_stats_dict in test_stats_dict['consumer_stats'].items()}
        test_stats.producer_stats = {index: ProducerStats.from_dict(producer_stats_dict, market) for index, producer_stats_dict in test_stats_dict['producer_stats'].items()}
        test_stats.influencer_stats = {index: InfluencerStats.from_dict(influencer_stats_dict, market) for index, influencer_stats_dict in test_stats_dict['influencer_stats'].items()}
        return test_stats