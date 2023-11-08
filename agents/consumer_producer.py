from agents.consumer import Consumer
from agents.producer import Producer
import numpy as np
from util import OptimizationTargets

class ConsumerProducer(Consumer, Producer):
    """
    A consumer-producer is an agent that can consume content and produce content.
    """
    
    def __init__(self, producer_topic_interest_function, consumer_topic_interest_function, attention_bound, external_interest_prob, delay_sensitivity, init_following_rates_method: str = 'random', optimize_tolerance = None):
        Consumer.__init__(self, consumer_topic_interest_function, attention_bound, external_interest_prob, delay_sensitivity, init_following_rates_method, optimize_tolerance)
        Producer.__init__(self, producer_topic_interest_function, optimize_tolerance)
    
    def set_market(self, market, index):
        Consumer.set_market(self, market, index)
        Producer.set_market(self, market, index)

    def reset(self, topic_position=None):
        Producer.reset(self, topic_position)
        Consumer.reset(self)

    def utility(self, x: np.array, *args) -> float:
        optimization_target = args[2]
        if optimization_target == OptimizationTargets.CONSUMER:
            return Consumer.utility(self, x, *args)
        elif optimization_target == OptimizationTargets.PRODUCER:
            return Producer.utility(self, x, *args)
        else:
            raise ValueError("Unknown optimization target.")

    def to_dict(self):
        consumer_producer_dict = Consumer.to_dict(self)
        producer_dict = Producer.to_dict(self)
        consumer_producer_dict.update(producer_dict)

        return consumer_producer_dict

    @staticmethod
    def from_dict(consumer_producer_dict: dict, market: 'ContentMarket'):
        consumer_producer = ConsumerProducer(lambda x: x, lambda x: x, consumer_producer_dict['attention_bound'], consumer_producer_dict['external_interest_prob'], consumer_producer_dict['delay_sensitivity'], consumer_producer_dict['init_following_rates_method'])
        consumer_producer.market = market
        consumer_producer.index = consumer_producer_dict['index']
        consumer_producer._following_rates = consumer_producer_dict['following_rates']
        consumer_producer.main_interest = np.array(consumer_producer_dict['main_interest'])
        consumer_producer.topic_produced = np.array(consumer_producer_dict['topic_produced'])

        return consumer_producer