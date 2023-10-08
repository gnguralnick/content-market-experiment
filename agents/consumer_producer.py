from consumer import Consumer
from producer import Producer
import numpy as np

class ConsumerProducer(Consumer, Producer):
    """
    A consumer-producer is an agent that can consume content and produce content.
    """
    
    def __init__(self, index: int, main_interest: np.ndarray, consumer_topic_interest_function, producer_topic_interest_function):
        Consumer.__init__(self, index,  main_interest, consumer_topic_interest_function)
        Producer.__init__(self, index, main_interest, producer_topic_interest_function)
    
    def set_market(self, market):
        Consumer.set_market(self, market)
        Producer.set_market(self, market)