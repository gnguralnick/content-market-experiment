from agents.consumer import Consumer
from agents.producer import Producer
import numpy as np
from util import OptimizationTargets

class ConsumerProducer(Consumer, Producer):
    """
    A consumer-producer is an agent that can consume content and produce content.
    """
    
    def __init__(self, producer_topic_interest_function, consumer_topic_interest_function, attention_bound, external_interest_prob, delay_sensitivity, init_following_rates_method: str = 'random'):
        Consumer.__init__(self, consumer_topic_interest_function, attention_bound, external_interest_prob, delay_sensitivity, init_following_rates_method)
        Producer.__init__(self, producer_topic_interest_function)
    
    def set_market(self, market, index):
        Consumer.set_market(self, market, index)
        Producer.set_market(self, market, index)

    def reset(self):
        Producer.reset(self)
        Consumer.reset(self)

    def utility(self, x: np.array, *args) -> float:
        optimization_target = args[2]
        if optimization_target == OptimizationTargets.CONSUMER:
            return Consumer.utility(self, x, *args)
        elif optimization_target == OptimizationTargets.PRODUCER:
            return Producer.utility(self, x, *args)
        else:
            raise ValueError("Unknown optimization target.")
