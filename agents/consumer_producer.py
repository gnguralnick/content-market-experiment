from consumer import Consumer
from producer import Producer
import numpy as np
from util import OptimizationTargets

class ConsumerProducer(Consumer, Producer):
    """
    A consumer-producer is an agent that can consume content and produce content.
    """
    
    def __init__(self, topic_interest_function):
        Consumer.__init__(self, topic_interest_function)
        Producer.__init__(self, topic_interest_function)
    
    def set_market(self, market):
        Consumer.set_market(self, market)
        Producer.set_market(self, market)

    def reset(self):
        Consumer.reset(self)
        Producer.reset(self)

    def utility(self, x: np.array, *args) -> float:
        optimization_target = args[2]
        if optimization_target == OptimizationTargets.CONSUMER:
            return Consumer.utility(self, x, *args)
        elif optimization_target == OptimizationTargets.PRODUCER:
            return Producer.utility(self, x, *args)
        else:
            raise ValueError("Unknown optimization target.")
