from agents.imperfect_producer import ImperfectInformationProducer
from agents.consumer_producer import ConsumerProducer
import numpy as np
from util import OptimizationTargets

class ImperfectConsumerProducer(ConsumerProducer, ImperfectInformationProducer):

    def __init__(self, topic_interest_function):
        ConsumerProducer.__init__(self, topic_interest_function)
        ImperfectInformationProducer.__init__(self, topic_interest_function)
    
    def utility(self, x: np.array, *args) -> float:
        optimization_target = args[2]
        if optimization_target == OptimizationTargets.CONSUMER:
            return ConsumerProducer.utility(self, x, *args)
        elif optimization_target == OptimizationTargets.PRODUCER:
            return ImperfectInformationProducer.utility(self, x, *args)
        else:
            raise ValueError("Unknown optimization target.")