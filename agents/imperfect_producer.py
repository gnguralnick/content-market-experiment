from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    from content_market import ContentMarket

import numpy as np
from agents.producer import Producer

class ImperfectInformationProducer(Producer):

    def __init__(self, index: int, main_interest: np.ndarray, topic_interest_function):
        super().__init__(index, main_interest, topic_interest_function)

    def utility(self, topic: np.ndarray, *args) -> float:
        producer = cast(Producer, args[0])
        if producer.market is None:
            raise ValueError("Producer has no market.")
        production_rate = cast(float, args[1])

        influencer_reward = 0
        for influencer in producer.market.influencers:
            # todo: recalculate influencer following rates
            influencer_reward += np.exp(-influencer.delay_sensitivity * (1 / influencer.producer_following_rates[producer.index]))
        
        return influencer_reward