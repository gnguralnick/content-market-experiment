from agents.imperfect_producer import ImperfectInformationProducer
from agents.consumer_producer import ConsumerProducer
import numpy as np
from util import OptimizationTargets

class ImperfectConsumerProducer(ConsumerProducer, ImperfectInformationProducer):

    def __init__(self, producer_topic_interest_function, consumer_topic_interest_function, attention_bound, external_interest_prob, delay_sensitivity, init_following_rates_method: str = 'random'):
        ConsumerProducer.__init__(self, producer_topic_interest_function, consumer_topic_interest_function, attention_bound, external_interest_prob, delay_sensitivity, init_following_rates_method)
        ImperfectInformationProducer.__init__(self, producer_topic_interest_function)
    
    def utility(self, x: np.array, *args) -> float:
        optimization_target = args[2]
        if optimization_target == OptimizationTargets.CONSUMER:
            return ConsumerProducer.utility(self, x, *args)
        elif optimization_target == OptimizationTargets.PRODUCER:
            return ImperfectInformationProducer.utility(self, x, *args)
        else:
            raise ValueError("Unknown optimization target.")