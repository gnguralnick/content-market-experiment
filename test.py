from agents import *
from content_market import ContentMarket
import numpy as np

def test(topics: np.ndarray, varied_param: str, num_agents: int | list[int], num_influencers: int | list[int],
         producer_topic_interest_func: Callable[[np.ndarray], float] | list[Callable[[np.ndarray], float]],
         consumer_topic_interest_func: Callable[[np.ndarray], float] | list[Callable[[np.ndarray], float]],
         consumer_attention_bound: float | list[float], consumer_external_interest_prob: float | list[float], consumer_delay_sensitivity: float | list[float],
         influencer_attention_bound: float | list[float], influencer_delay_sensitivity: float | list[float],
         init_following_rates_method: str = 'random', init_interest_method: str = 'random',
         production_rate: float | list[float] = 1, external_production_rate: float | list[float] = 1,
         use_imperfect_information: bool = False):
    """
    Test the market with the given parameters.
    A test will be run for every combination of parameters.
    """

    # create a list of all parameters
    param_names = ['num_agents', 'num_influencers', 'producer_topic_interest_func', 'consumer_topic_interest_func',
                   'consumer_attention_bound', 'consumer_external_interest_prob', 'consumer_delay_sensitivity',
                   'influencer_attention_bound', 'influencer_delay_sensitivity', 'production_rate', 'external_production_rate']
    params = [num_agents, num_influencers, producer_topic_interest_func, consumer_topic_interest_func,
                consumer_attention_bound, consumer_external_interest_prob, consumer_delay_sensitivity,
                influencer_attention_bound, influencer_delay_sensitivity, production_rate, external_production_rate]
    
    # find the index of the varied parameter
    varied_param_index = param_names.index(varied_param)
    if not isinstance(params[varied_param_index], list):
        print('WARNING: varied_param is not a list. It will be converted to a list.')
        params[varied_param_index] = [params[varied_param_index]]

    # create a list of all combinations of parameters
    param_combinations = []
    for i in range(len(params[varied_param_index])):
        combo = [params[j] if j != varied_param_index else params[j][i] for j in range(len(params))]
        param_combinations.append(combo)

    perfect_info_stats = []
    imperfect_info_stats = []

    # run the test for each combination of parameters
    for combo in param_combinations:
        [agents, influencers, prod_topic_interest_func, cons_topic_interest_func,
         cons_att_bound, cons_ext_prob, cons_delay_sens, influencer_att_bound, influencer_delay_sens,
         prod_rate, ext_prod_rate] = combo
        # create the market
        market = ContentMarket(topics, prod_rate, ext_prod_rate)
        for i in range(agents):
            market.add_agent(ConsumerProducer(topic_interest_func, cons_att_bound, cons_ext_prob, cons_delay_sens, init_following_rates_method))
        for i in range(influencers):
            market.add_agent(Influencer(influencer_att_bound, influencer_delay_sens, init_following_rates_method))
        
        # initialize the market
        market.finalize(method=init_interest_method)
        # run the market
        test_stats = market.optimize()
        perfect_info_stats.append(test_stats)
    if use_imperfect_information:
        for combo in param_combinations:
            [agents, influencers, topic_interest_func,
            cons_att_bound, cons_ext_prob, cons_delay_sens, influencer_att_bound, influencer_delay_sens,
            prod_rate, ext_prod_rate] = combo
            # create the market
            market = ContentMarket(topics, prod_rate, ext_prod_rate)
            for i in range(agents):
                market.add_agent(ImperfectConsumerProducer(topic_interest_func, cons_att_bound, cons_ext_prob, cons_delay_sens, init_following_rates_method))
            for i in range(influencers):
                market.add_agent(Influencer(influencer_att_bound, influencer_delay_sens, init_following_rates_method))
            
            # initialize the market
            market.finalize(method=init_interest_method)
            # run the market
            test_stats = market.optimize()
            imperfect_info_stats.append(test_stats)
            
    return perfect_info_stats, imperfect_info_stats

