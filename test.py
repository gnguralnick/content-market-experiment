from agents import *
from content_market import ContentMarket
import numpy as np

def test(topics: np.ndarray, varied_param: str, num_producers: int | list[int], num_influencers: int | list[int], num_consumers: int | list[int],
         producer_topic_func: Callable[[np.ndarray], float] | list[Callable[[np.ndarray], float]], consumer_topic_func: Callable[[np.ndarray], float] | list[Callable[[np.ndarray], float]],
         consumer_attention_bound: float | list[float], consumer_external_interest_prob: float | list[float], consumer_delay_sensitivity: float | list[float],
         influencer_attention_bound: float | list[float], influencer_delay_sensitivity: float | list[float],
         init_following_rates_method: str = 'random', init_interest_method: str = 'random',
         production_rate: float | list[float] = 1, external_production_rate: float | list[float] = 1,
         producer_type=Producer, consumer_type=Consumer, influencer_type=Influencer):
    """
    Test the market with the given parameters.
    A test will be run for every combination of parameters.
    """
    # use varied_param to determine which parameter to vary

    # create a list of all parameters
    param_names = ['num_producers', 'num_influencers', 'num_consumers', 'producer_topic_func', 'consumer_topic_func',
                   'consumer_attention_bound', 'consumer_external_interest_prob', 'consumer_delay_sensitivity',
                   'influencer_attention_bound', 'influencer_delay_sensitivity', 'production_rate', 'external_production_rate']
    params = [num_producers, num_influencers, num_consumers, producer_topic_func, consumer_topic_func,
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

    stats = {
        'total_social_welfare': [],
        'average_agent_utilities': [],
        'average_producer_utilities': [],
        'average_influencer_utilities': [],
        'average_consumer_utilities': [],
        'num_iterations': [],
        'average_consumer_following_rate_influencer_proportion': [],
    }
    markets = []
    tests = { varied_param: params[varied_param_index] }

    # run the test for each combination of parameters
    for combo in param_combinations:
        [producers, influencers, consumers, prod_topic_func, cons_topic_func,
         cons_att_bound, cons_ext_prob, cons_delay_sens, influencer_att_bound, influencer_delay_sens,
         prod_rate, ext_prod_rate] = combo
        # create the market
        market = ContentMarket(topics)
        markets.append(market)
        # create the agents
        for i in range(producers):
            market.add_agent(producer_type(prod_topic_func))
        for i in range(influencers):
            market.add_agent(influencer_type(influencer_att_bound, influencer_delay_sens, init_following_rates_method))
        for i in range(consumers):
            market.add_agent(consumer_type(cons_topic_func, cons_att_bound, cons_ext_prob, cons_delay_sens, init_following_rates_method))
        # initialize the market
        market.finalize(method=init_interest_method)
        # run the market
        consumer_stats, producer_stats, influencer_stats, total_stats, average_stats = market.optimize(production_rate=prod_rate, external_production_rate=ext_prod_rate)

        # add the stats to the list
        stats['total_social_welfare'].append(total_stats['social_welfare'])
        stats['average_agent_utilities'].append(average_stats['agent_utilities'])
        stats['average_producer_utilities'].append(average_stats['producer_utilities'])
        stats['average_influencer_utilities'].append(average_stats['influencer_utilities'])
        stats['average_consumer_utilities'].append(average_stats['consumer_utilities'])
        stats['num_iterations'].append(len(total_stats['social_welfare']))

        # average proportion of consumer attention allocated to influencers
        influencer_proportion_sum = 0
        for consumer in market.consumers:
            for influencer in market.influencers:
                influencer_proportion_sum += consumer.following_rates[influencer.index] / consumer.attention_bound
            influencer_proportion_sum /= len(market.influencers)
        influencer_proportion_sum /= len(market.consumers)
        stats['average_consumer_following_rate_influencer_proportion'].append(influencer_proportion_sum)
            

    return stats, markets, tests

