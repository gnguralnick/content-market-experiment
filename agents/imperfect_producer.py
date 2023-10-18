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
        
        for influencer in self.market.influencers:
            prev_rate = influencer.get_following_rate_vector()
            
            attention_constraint = LinearConstraint(np.ones(self.market.num_agents + 1), lb=0, ub=influencer.attention_bound)

            bounds = []
            for agent in self.market.agents:
                if agent == influencer or (not isinstance(agent, Producer) and not isinstance(agent, Influencer)):
                    bounds.append((0, 0))
                else:
                    bounds.append((0, None))
            bounds.append((influencer.get_following_rate_vector()[-1], influencer.get_following_rate_vector()[-1])) # external should not change when optimizing influencer

            result = minimize(
                fun=influencer.minimization_utility,
                x0=influencer.get_following_rate_vector(),
                args=(production_rate, external_production_rate, OptimizationTargets.INFLUENCER),
                constraints=attention_constraint,
                bounds=bounds
            )

            if not result.success:
                raise RuntimeError("Optimization failed", result)
            
        influencer_reward = 0
        for influencer in self.market.influencers:

            influencer_reward += np.exp(-influencer.delay_sensitivity * (1 / influencer.following_rates[self.index]))

            influencer.set_following_rate_vector(prev_rate)
        
        self.topic_produced = prev_topic
        return influencer_reward