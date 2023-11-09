from agents import *
from content_market import ContentMarket
import numpy as np
from typing import Sequence

def test(topics: np.ndarray | Sequence[np.ndarray], varied_param: str, num_agents: int | Sequence[int], num_influencers: int | Sequence[int],
         producer_topic_interest_func: Callable[[np.ndarray], float] | Sequence[Callable[[np.ndarray], float]] | None,
         consumer_topic_interest_func: Callable[[np.ndarray], float] | Sequence[Callable[[np.ndarray], float]] | None,
         agent_topic_interest_func: Callable[[np.ndarray], float] | Sequence[Callable[[np.ndarray], float]] | None,
         consumer_attention_bound: float | Sequence[float], consumer_external_interest_prob: float | Sequence[float], 
         delay_sensitivity: float | Sequence[float] | None, consumer_delay_sensitivity: float | Sequence[float] | None,
         influencer_attention_bound: float | Sequence[float], influencer_delay_sensitivity: float | Sequence[float] | None,
         init_following_rates_method: str | Sequence[str] = 'random', init_interest_method: str = 'random',
         init_topic_produced_method: str | Sequence[str] = 'main',
         production_rate: float | Sequence[float] = 1, external_production_rate: float | Sequence[float] = 1,
         use_imperfect_information: bool = False):
    """
    Test the market with the given parameters.
    A test will be run for every combination of parameters.
    """

    # create a list of all parameters
    param_names = ['topics', 'num_agents', 'num_influencers', 'producer_topic_interest_func', 'consumer_topic_interest_func', 'agent_topic_interest_func',
                   'consumer_attention_bound', 'consumer_external_interest_prob', 'delay_sensitivity', 'consumer_delay_sensitivity',
                   'influencer_attention_bound', 'influencer_delay_sensitivity', 'production_rate', 'external_production_rate', 
                   'init_following_rates_method', 'init_topic_produced_method']
    params = [topics, num_agents, num_influencers, producer_topic_interest_func, consumer_topic_interest_func, agent_topic_interest_func,
                consumer_attention_bound, consumer_external_interest_prob, delay_sensitivity, consumer_delay_sensitivity, 
                influencer_attention_bound, influencer_delay_sensitivity, production_rate, external_production_rate, 
                init_following_rates_method, init_topic_produced_method]
    
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
        [topics_bounds, agents, influencers, prod_topic_interest_func, cons_topic_interest_func, ag_topic_interest_func,
         cons_att_bound, cons_ext_prob, delay_sens, cons_delay_sens, influencer_att_bound, influencer_delay_sens,
         prod_rate, ext_prod_rate, following_rates_init, topic_produced_init] = combo
        # create the market
        market = ContentMarket(topics_bounds, prod_rate, ext_prod_rate)
        c_delay_sens = delay_sens if cons_delay_sens is None else cons_delay_sens
        i_delay_sens = delay_sens if influencer_delay_sens is None else influencer_delay_sens
        for i in range(agents):
            if ag_topic_interest_func is None:
                market.add_agent(ConsumerProducer(prod_topic_interest_func, cons_topic_interest_func, cons_att_bound, cons_ext_prob, c_delay_sens, following_rates_init))
            else:
                market.add_agent(ConsumerProducer(ag_topic_interest_func, ag_topic_interest_func, cons_att_bound, cons_ext_prob, c_delay_sens, following_rates_init))
        for i in range(influencers):
            market.add_agent(Influencer(influencer_att_bound, i_delay_sens, following_rates_init))
        
        # initialize the market
        market.finalize(method=init_interest_method)
        # run the market
        test_stats = market.optimize(topic_position=topic_produced_init)
        perfect_info_stats.append(test_stats)
    if use_imperfect_information:
        for combo in param_combinations:
            [topics_bounds, agents, influencers, prod_topic_interest_func, cons_topic_interest_func, ag_topic_interest_func,
            cons_att_bound, cons_ext_prob, delay_sens, cons_delay_sens, influencer_att_bound, influencer_delay_sens,
            prod_rate, ext_prod_rate, following_rates_init, topic_produced_init] = combo
            # create the market
            market = ContentMarket(topics_bounds, prod_rate, ext_prod_rate)
            c_delay_sens = delay_sens if cons_delay_sens is None else cons_delay_sens
            i_delay_sens = delay_sens if influencer_delay_sens is None else influencer_delay_sens
            for i in range(agents):
                if ag_topic_interest_func is None:
                    market.add_agent(ImperfectConsumerProducer(prod_topic_interest_func, cons_topic_interest_func, cons_att_bound, cons_ext_prob, c_delay_sens, following_rates_init))
                else:
                    market.add_agent(ImperfectConsumerProducer(ag_topic_interest_func, ag_topic_interest_func, cons_att_bound, cons_ext_prob, c_delay_sens, following_rates_init))
            for i in range(influencers):
                market.add_agent(Influencer(influencer_att_bound, i_delay_sens, following_rates_init))
            
            # initialize the market
            market.finalize(method=init_interest_method)
            # run the market
            test_stats = market.optimize(topic_position=topic_produced_init)
            imperfect_info_stats.append(test_stats)
            
    return perfect_info_stats, imperfect_info_stats