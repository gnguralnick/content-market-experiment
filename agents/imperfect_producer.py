from typing import cast

from agents.influencer import Influencer

import numpy as np
from agents.producer import Producer

from scipy.optimize import minimize, LinearConstraint

from util import OptimizationTargets

class ImperfectInformationProducer(Producer):

    def __init__(self, topic_interest_function):
        super().__init__(topic_interest_function)
        self.use_imperfect_information = True

    def toggle_imperfect_information(self):
        self.use_imperfect_information = not self.use_imperfect_information

    def utility(self, topic: np.ndarray, *args) -> float:
        if not self.use_imperfect_information:
            return super().utility(topic, *args)
        if self.market is None:
            raise ValueError("Producer has no market.")
        production_rate = cast(float, args[0])
        external_production_rate = cast(float, args[1])

        prev_topic = self.topic_produced
        self.topic_produced = topic

        prev_rates = {influencer.index: influencer.get_following_rate_vector() for influencer in self.market.influencers}
        potential_rates = {}

        for influencer in self.market.influencers:
            attention_constraint = LinearConstraint(np.ones(self.market.num_agents + 1), lb=0, ub=influencer.attention_bound)
            
            print("Optimizing for influencer", influencer.index, "under imperfect producer", self.index)
            result = minimize(
                fun=influencer.minimization_utility,
                x0=influencer.get_following_rate_vector(),
                args=(production_rate, external_production_rate, OptimizationTargets.INFLUENCER),
                constraints=attention_constraint,
                bounds=influencer.get_following_rate_bounds(),
                options={'maxiter': 1000},
                tol=1e-15,
            )

            if not result.success:
                print(result)
                raise RuntimeError("Optimization failed", result.message)
    
            
            potential_rates[influencer.index] = result.x
            
        influencer_reward = 0
        old_reward = 0
        for influencer in self.market.influencers:
            influencer_reward += np.exp(-influencer.delay_sensitivity * (1 / potential_rates[influencer.index][self.index]))
            old_reward += np.exp(-influencer.delay_sensitivity * (1 / prev_rates[influencer.index][self.index]))

        
        self.topic_produced = prev_topic
        return influencer_reward