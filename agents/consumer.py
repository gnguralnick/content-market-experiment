import numpy as np
from typing import cast, Callable, TYPE_CHECKING
from agents.agent import Agent
from agents.producer import Producer
from agents.influencer import Influencer
if TYPE_CHECKING:
    from content_market import ContentMarket


class Consumer(Agent):
    def __init__(self, topic_interest_function: Callable[[float], float], attention_bound, external_interest_prob, delay_sensitivity, init_following_rates_method: str = 'random', optimize_tolerance = None):
        Agent.__init__(self, optimize_tolerance)
        self.main_interest = None

        self._consumption_topic_interest_function = topic_interest_function
        self.attention_bound = attention_bound
        
        self.external_interest_prob = external_interest_prob
        self.delay_sensitivity = delay_sensitivity

        self.init_following_rates_method = init_following_rates_method

    def init_following_rates(self):
        Agent.init_following_rates(self)
        if self.init_following_rates_method == 'random':
            cur_sum = 0
            agent_random_sort = np.random.permutation(self.market.agents)
            for agent in agent_random_sort:
                if agent == self or (not isinstance(agent, Producer) and not isinstance(agent, Influencer)):
                    # only producers and influencers can be followed
                    continue
                self._following_rates[agent.index] = np.random.uniform(0, self.attention_bound - cur_sum)
                cur_sum += self._following_rates[agent.index]
            
            self._following_rates['external'] = np.random.uniform(0, self.attention_bound - cur_sum)
        elif self.init_following_rates_method == 'equal':
            num_producers_or_influencers = len(set(self.market.producers + self.market.influencers) - {self})
            for agent in self.market.agents:
                if agent == self or (not isinstance(agent, Producer) and not isinstance(agent, Influencer)):
                    # only producers and influencers can be followed
                    continue
                self._following_rates[agent.index] = self.attention_bound / (num_producers_or_influencers + 1) # +1 for external
            
            self._following_rates['external'] = self.attention_bound / (num_producers_or_influencers + 1)
        elif self.init_following_rates_method == 'zero':
            self._following_rates = {agent.index: 1e-1 for agent in self.market.agents}
            self._following_rates['external'] = 1e-1
        else:
            raise ValueError("Unknown init_following_rates_method.")

    def set_main_interest(self, main_interest: np.ndarray):
        self.main_interest = main_interest
    
    def reset(self):
        return Agent.reset(self)

    def consumption_topic_interest(self, topic: np.ndarray) -> float:
        if not self.market.check_topic(topic):
            raise ValueError("Topic is not in the market.")
        distance = np.linalg.norm(topic - self.main_interest)
        return self._consumption_topic_interest_function(distance)
    
    def check_following_rates(self, value: dict[int, float]) -> bool:
        if sum(value.values()) - self.attention_bound > 1e-6:
            return False
        return True
    
    def get_following_rate_bounds(self):
        bounds = []
        for agent in self.market.agents:
            if agent == self:
                bounds.append((0, 0))
            elif not isinstance(agent, Producer) and not isinstance(agent, Influencer):
                bounds.append((self.following_rates[agent.index], self.following_rates[agent.index]))
            else:
                bounds.append((0, None))
        bounds.append((0, None)) # external
        return bounds

    def utility(self, x: np.array, *args) -> float:
        if self.market is None or self.index is None:
            raise ValueError("Consumer has no market.")
        production_rate: float = cast(float, args[0])
        external_production_rate: float = cast(float, args[1])

        following_rate_vector = x
        
        influencer_reward = 0
        for influencer in self.market.influencers:
            if self == influencer:
                continue
            for producer in self.market.producers:
                if not influencer.following_rates[producer.index] > 0:
                    continue
                if not following_rate_vector[influencer.index] > 0:
                    continue

                if self == producer:
                    continue

                topic_reward = producer.topic_probability(producer.topic_produced) * self.consumption_topic_interest(producer.topic_produced)
                delay = np.exp(-self.delay_sensitivity * (1 / influencer.following_rates[producer.index] + 1 / following_rate_vector[influencer.index]))

                influencer_reward += production_rate * topic_reward * delay
        
        direct_following_reward = 0
        for producer in self.market.producers:
            if not following_rate_vector[producer.index] > 0:
                continue

            if self == producer:
                continue

            topic_reward = producer.topic_probability(producer.topic_produced) * self.consumption_topic_interest(producer.topic_produced)
            delay = np.exp(-self.delay_sensitivity * (1 / following_rate_vector[producer.index]))
            direct_following_reward += production_rate * topic_reward * delay

        external_reward = 0
        if following_rate_vector[-1] > 0:
            external_reward = external_production_rate * self.external_interest_prob * np.exp(-self.delay_sensitivity * (1 / following_rate_vector[-1]))

        return influencer_reward + direct_following_reward + external_reward
    
    def to_dict(self) -> dict:
        consumer_dict = Agent.to_dict(self)
        consumer_dict['main_interest'] = self.main_interest.tolist()
        consumer_dict['attention_bound'] = self.attention_bound
        consumer_dict['external_interest_prob'] = self.external_interest_prob
        consumer_dict['delay_sensitivity'] = self.delay_sensitivity
        consumer_dict['init_following_rates_method'] = self.init_following_rates_method
        return consumer_dict

    @staticmethod
    def from_dict(consumer_dict: dict, market: 'ContentMarket'):
        consumer = Consumer(lambda x: x, consumer_dict['attention_bound'], consumer_dict['external_interest_prob'], consumer_dict['delay_sensitivity'], consumer_dict['init_following_rates_method'])
        consumer.main_interest = np.array(consumer_dict['main_interest'])
        consumer.market = market
        market.add_agent(consumer)
        consumer.index = consumer_dict['index']
        consumer.following_rates = consumer_dict['_following_rates']
        consumer.optimize_tolerance = consumer_dict['optimize_tolerance']
        return consumer
