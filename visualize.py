from typing import Sequence
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
palette = list(mcd.XKCD_COLORS.values())[::10]
import numpy as np

from agents import Consumer, Producer, Agent, Influencer
from stats import *
from content_market import ContentMarket

import networkx as nx

def get_agent_title(agent: Agent):
    subclass_checks = [isinstance(agent, Consumer), isinstance(agent, Producer), isinstance(agent, Influencer)]
    if not any(subclass_checks) or sum(subclass_checks) > 1:
        return "Agent {}".format(agent.index)

    if isinstance(agent, Producer):
        return "Producer {}".format(agent.index)
    elif isinstance(agent, Consumer):
        return "Consumer {}".format(agent.index)
    elif isinstance(agent, Influencer):
        return "Influencer {}".format(agent.index)

def plot_topic_distribution_histogram(title, agents: Sequence[Consumer | Producer], min_topic, max_topic, bins=50):
    if len(agents) == 0:
        return
    plt.figure()
    plt.title(title)
    plt.xlabel('Topic')
    plt.ylabel('Frequency')
    plt.hist([agent.main_interest[0] for agent in agents], bins=bins)
    plt.yticks(range(0, len(agents) + 1))
    plt.xlim(min_topic, max_topic)
    plt.show()

def plot_consumer_topic_interest_distributions(title, agents: Sequence[Consumer], min_topic, max_topic, agent_colors):
    if len(agents) == 0:
        return
    plt.figure()
    plt.title(title)
    topics_dist = np.linspace(min_topic, max_topic, 100)
    for a in agents:
        plt.plot(topics_dist, [a.consumption_topic_interest(np.array(t).reshape((1,))) for t in topics_dist], color=agent_colors[a.index], label=f"Consumer {a.index}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(0, 1.1)
    plt.show()

def plot_producer_topic_probability_distributions(title, agents: Sequence[Producer], min_topic, max_topic, agent_colors):
    if len(agents) == 0:
        return
    plt.figure()
    plt.title(title)
    topics_dist = np.linspace(min_topic, max_topic, 100)
    for a in agents:
        plt.plot(topics_dist, [a.topic_probability(np.array(t).reshape((1,))) for t in topics_dist], color=agent_colors[a.index], label=f"Producer {a.index}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(0, 1.1)
    plt.show()

def plot_agent_utility_by_iteration(title, agents: Sequence[Agent], agent_colors, agent_stats: dict[int, AgentStats], averages=None):
    if len(agents) == 0:
        return
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for agent in agents:
        plt.plot(agent_stats[agent.index].utilities, label=get_agent_title(agent), color=agent_colors[agent.index])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(0, len(agent_stats[agents[0].index].utilities)))
    plt.show()

def plot_agent_utility_change_by_iteration(title, agents: Sequence[Agent], agent_colors, agent_stats: dict[int, AgentStats], averages=None):
    if len(agents) == 0:
        return
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for agent in agents:
        plt.plot(agent_stats[agent.index].utility_change, label=get_agent_title(agent), color=agent_colors[agent.index])
    plt.legend()
    plt.xticks(range(0, len(agent_stats[agents[0].index].utility_change)))
    plt.show()

def plot_attention_used_by_iteration(title, agents: Sequence[Consumer | Influencer], agent_colors, agent_stats: dict[int, ConsumerStats | InfluencerStats], averages=None):
    if len(agents) == 0:
        return
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for agent in agents:
        plt.plot(list(range(0, len(agent_stats[agent.index].attention_used))), agent_stats[agent.index].attention_used, label=get_agent_title(agent), color=agent_colors[agent.index])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(0, len(agent_stats[agents[0].index].attention_used)))
    plt.ylim(0, max(agent.attention_bound for agent in agents) + 1)
    plt.show()

def plot_producer_topic_produced_by_iteration(title, producers: Sequence[Producer], consumers: Sequence[Consumer], agent_colors, agent_stats: dict[int, ProducerStats], show_consumer_main_interest=True):
    if len(producers) == 0:
        return
    plt.figure()
    plt.title(title)
    for producer in producers:
        plt.plot(agent_stats[producer.index].topics, label='Producer {}'.format(producer.index), color=agent_colors[producer.index])
    if show_consumer_main_interest:
        for consumer in consumers:
            plt.plot([consumer.main_interest] * len(agent_stats[producer.index].topics), label='Consumer {}'.format(consumer.index), linestyle='--', color=agent_colors[consumer.index])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(min(min(agent.main_interest for agent in producers), min(agent.main_interest for agent in consumers)) - 0.05, max(max(agent.main_interest for agent in producers), max(agent.main_interest for agent in consumers)) + 0.05)
    plt.xticks(range(len(agent_stats[producers[0].index].topics)))
    plt.show()

def plot_following_rate_change_by_iteration(title, agents: Sequence[Consumer | Influencer], agent_colors, agent_stats: dict[int, ConsumerStats | InfluencerStats], averages=None):
    if len(agents) == 0:
        return
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for agent in agents:
        plt.plot(agent_stats[agent.index].rate_change, label=get_agent_title(agent), color=agent_colors[agent.index])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(0, len(agent_stats[agents[0].index].rate_change)))
    plt.show()

def plot_producer_topic_change_by_iteration(title, producers: Sequence[Producer], agent_colors, agent_stats: dict[int, ProducerStats], averages=None):
    if len(producers) == 0:
        return
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for producer in producers:
        plt.plot(agent_stats[producer.index].topic_change, label='Producer {}'.format(producer.index), color=agent_colors[producer.index])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(len(agent_stats[producers[0].index].topic_change)))
    plt.show()

def plot_total_social_welfare_by_iteration(title, stats: TestStats):
    plt.figure()
    plt.title(title)
    plt.plot(stats.total_social_welfare)
    plt.xticks(range(0, stats.num_iterations))
    plt.show()

def plot_following_rate_by_main_interest_closeness(title, consumers: Sequence[Consumer], producers: Sequence[Producer], agent_colors, agent_stats: dict[int, ConsumerStats], averages=None):
    if len(consumers) == 0 or len(producers) == 0:
        return
    plt.figure()
    plt.title(title)
    plt.xlabel('Main interest closeness')
    plt.ylabel('Following rate (%)')
    if averages:
        plt.plot(averages[0], averages[1], label="Average")
    for consumer in consumers:
        interest_closeness = []
        following_rate = []
        for producer in sorted(producers, key=lambda p: np.linalg.norm(consumer.main_interest - p.main_interest)):
            if producer == consumer:
                continue
            ending_rate = agent_stats[consumer.index].following_rates[-1][producer.index] / consumer.attention_bound * 100
            interest_closeness.append(np.linalg.norm(consumer.main_interest - producer.main_interest))
            following_rate.append(ending_rate)
        plt.plot(interest_closeness, following_rate, label='Consumer {}'.format(consumer.index), color=agent_colors[consumer.index], marker='o')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plot_following_rates_by_iteration(agents: Sequence[Consumer | Influencer], follows: Sequence[Producer | Influencer], agent_colors, agent_stats: dict[int, ConsumerStats | InfluencerStats]):
    if len(agents) == 0 or len(follows) == 0:
        return
    fig = plt.figure(figsize=(5, 5 * len(agents)))
    for i, agent in enumerate(agents):
        ax = fig.add_subplot(len(agents), 1, i + 1)
        ax.set_title(f"Following rates by Iteration for {get_agent_title(agent)}")
        for other in follows:
            if other == agent:
                continue
            rate_by_iteration = [vec[other.index] for vec in agent_stats[agent.index].following_rates]
            ax.plot(rate_by_iteration, label=get_agent_title(other), color=agent_colors[other.index])
        if isinstance(agent, Consumer):
            rate_by_iteration = [vec[-1] for vec in agent_stats[agent.index].following_rates]
            ax.plot(rate_by_iteration, label='External', color='black')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_xticks(range(len(rate_by_iteration)))
        ax.sharex(fig.axes[0])
        ax.label_outer()
    plt.show()

def plot_agent_following_rates(agents: Sequence[Consumer | Influencer], agent_stats: dict[int, ConsumerStats | InfluencerStats], agent_colors):
    if len(agents) == 0:
        return
    fig = plt.figure(figsize=(5, 5 * len(agents)))
    for i, agent in enumerate(agents):
        ax = fig.add_subplot(len(agents), 1, i + 1)
        ax.set_title(f"Following rates percentages for {get_agent_title(agent)}")
        topics_bounds = agent.market.topics_bounds[0]
        
        prod_main_interest_with_rates = [(prod.main_interest[0], agent_stats[agent.index].following_rates[-1][prod.index] / agent.attention_bound * 100) for prod in agent.market.producers if prod != agent]
        prod_main_interest_with_rates.sort(key=lambda x: x[0])
        ax.plot([x[0] for x in prod_main_interest_with_rates], [x[1] for x in prod_main_interest_with_rates], label='Following rate by producer main interest', marker='o')

        prod_topic_produced_with_rates = [(prod.topic_produced[0], agent_stats[agent.index].following_rates[-1][prod.index] / agent.attention_bound * 100) for prod in agent.market.producers if prod != agent]
        prod_topic_produced_with_rates.sort(key=lambda x: x[0])
        ax.plot([x[0] for x in prod_topic_produced_with_rates], [x[1] for x in prod_topic_produced_with_rates], label='Following rate by producer topic produced', marker='o')
        
        if isinstance(agent, Consumer):
            ax.axvline(agent.main_interest[0], color='black', linestyle='--', label='Consumer main interest')
            #ax.axhline(agent_stats[agent.index].following_rates[-1][-1] / agent.attention_bound * 100, color='black', linestyle='--', label='External')
            #ax.axhline(agent_stats[agent.index].following_rates[-1][-2] / agent.attention_bound * 100, color='blue', linestyle='--', label='Influencer')

        
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_xlim(topics_bounds[0], topics_bounds[1])
        ax.set_xlabel('Topic')
    plt.show()

def plot_follows_by_iteration(agents: Sequence[Producer | Influencer], followers: Sequence[Consumer | Influencer], agent_colors, agent_stats: dict[int, ConsumerStats | InfluencerStats]):
    if len(agents) == 0 or len(followers) == 0:
        return
    fig = plt.figure(figsize=(5, 5 * len(agents)))
    for i, agent in enumerate(agents):
        ax = fig.add_subplot(len(agents), 1, i + 1)
        ax.set_title(f"{get_agent_title(agent)} followers by Iteration")
        for follower in followers:
            if follower == agent:
                continue
            following_rate_by_iteration = [vec[agent.index] for vec in agent_stats[follower.index].following_rates]
            ax.plot(following_rate_by_iteration, label=get_agent_title(follower), color=agent_colors[follower.index])
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_xticks(range(len(following_rate_by_iteration)))
        ax.sharex(fig.axes[0])
        ax.label_outer()
    plt.show()

def plot_ending_value_by_test(title, perfect_info_stats: list[TestStats], imperfect_info_stats: list[TestStats], value_name, varied_values, xlabel, ylabel, figure=True):
    if figure:
        plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(perfect_info_stats) > 0:
        perfect_ending_values = [getattr(perfect_info_stats[i], value_name)[-1] for i in range(len(perfect_info_stats))]
        plt.plot(varied_values, perfect_ending_values, label='Perfect Information')
    if len(imperfect_info_stats) > 0:
        imperfect_ending_values = [getattr(imperfect_info_stats[i], value_name)[-1] for i in range(len(imperfect_info_stats))]
        plt.plot(varied_values, imperfect_ending_values, label='Imperfect Information')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim(min(varied_values), max(varied_values))
    plt.show()

def plot_agent_following_rates_by_test(index: int, stats: Sequence[dict[int, ConsumerStats | InfluencerStats]], varied_param, varied_values):
    plt.figure()
    plt.title(f"Following rates for {get_agent_title(stats[0][index].agent)}")
    topics_bounds = stats[0][index].agent.market.topics_bounds[0]

    agent = stats[0][index].agent
    if isinstance(agent, Consumer):
        plt.axvline(agent.main_interest[0], color='black', linestyle='--', label='Consumer main interest')

    for i in range(len(stats)):
        agent_stats = stats[i][index]
        agent: Consumer | Influencer = agent_stats.agent
        prod_main_interest_with_rates = [(prod.main_interest[0], agent_stats.following_rates[-1][prod.index] / agent.attention_bound * 100) for prod in agent.market.producers if prod != agent]
        prod_main_interest_with_rates.sort(key=lambda x: x[0])
        plt.plot([x[0] for x in prod_main_interest_with_rates], [x[1] for x in prod_main_interest_with_rates], label=f'{varied_param} = {varied_values[i]}', marker='o', color=palette[i])


    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim(topics_bounds[0] - 0.05, topics_bounds[1] + 0.05)
    plt.xlabel('Producer Main Interest')
    plt.ylabel('Following rate (%)')
    plt.show()


def plot_cost_of_influence_by_test(title, perfect_info_stats: list[TestStats], imperfect_info_stats: list[TestStats], varied_values, xlabel, figure=True):
    if figure:
        plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Cost of influence (%)')
    cost = []
    if not len(perfect_info_stats) == len(imperfect_info_stats):
        raise ValueError("Perfect and imperfect information stats must have the same length.")
    for i in range(len(perfect_info_stats)):
        if perfect_info_stats[i].total_social_welfare[-1] == 0:
            cost.append(0)
            continue
        cost.append((perfect_info_stats[i].total_social_welfare[-1] - imperfect_info_stats[i].total_social_welfare[-1]) / perfect_info_stats[i].total_social_welfare[-1] * 100)
    plt.plot(varied_values, cost)
    plt.xlim(min(varied_values), max(varied_values))
    plt.show()

def plot_value_by_test(title, perfect_info_stats: list[TestStats], imperfect_info_stats: list[TestStats], value_name, varied_values, xlabel, ylabel, figure=True):
    if figure:
        plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(perfect_info_stats) > 0:
        perfect_values = [getattr(perfect_info_stats[i], value_name) for i in range(len(perfect_info_stats))]
        plt.plot(varied_values, perfect_values, label='Perfect Information')
    if len(imperfect_info_stats) > 0:
        imperfect_values = [getattr(imperfect_info_stats[i], value_name) for i in range(len(imperfect_info_stats))]
        plt.plot(varied_values, imperfect_values, label='Imperfect Information')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim(min(varied_values), max(varied_values))
    plt.show()

def plot_value_by_iteration(title, stats: TestStats, value_name, ylabel):
    plt.figure()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.plot(getattr(stats, value_name))
    plt.xticks(range(stats.num_iterations))
    plt.show()

def plot_value_by_agent_by_iteration(title, agent_stats: dict[int, AgentStats], value_name, ylabel, averages=None):
    plt.figure()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    if averages:
        plt.plot(averages, label="Average")
    for agent in agent_stats:
        plt.plot(getattr(agent_stats[agent], value_name), label=get_agent_title(agent_stats[agent].agent))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(len(getattr(agent_stats[0], value_name))))
    plt.show()

def plot_value_by_iteration_by_test(title, stats: list[TestStats], value_name, varied_name, varied_values, ylabel, figure=True):
    if figure:
        plt.figure()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    for i in range(len(stats)):
        plt.plot(getattr(stats[i], value_name), label=f"{varied_name} = {varied_values[i]}", color=palette[i])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(max(test.num_iterations for test in stats)))
    plt.show()

def plot_producer_topic_distance_from_main_interest_by_iteration(title, producers: Sequence[Producer], agent_colors, agent_stats: dict[int, ProducerStats], averages=None):
    if len(producers) == 0:
        return
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for producer in producers:
        plt.plot(agent_stats[producer.index].topic_distance, label='Producer {}'.format(producer.index), color=agent_colors[producer.index])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(len(agent_stats[producers[0].index].topic_distance)))
    plt.show()
    

def plot_follow_proportion_by_iteration(title, consumers: Sequence[Consumer], agent_colors, agent_stats: dict[int, ConsumerStats], agent: str, averages=None):
    if len(consumers) == 0:
        return
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for consumer in consumers:
        plt.plot(agent_stats[consumer.index].get_follow_proportion(agent), label='Consumer {}'.format(consumer.index), color=agent_colors[consumer.index])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(len(agent_stats[consumers[0].index].get_follow_proportion(agent))))
    plt.show()

def plot_ending_value_perfect_imperfect_difference_by_test(title, perfect_info_stats: list[TestStats], imperfect_info_stats: list[TestStats], value_name, varied_values, xlabel, ylabel):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(perfect_info_stats) > 0:
        perfect_ending_values = [getattr(perfect_info_stats[i], value_name)[-1] for i in range(len(perfect_info_stats))]
    if len(imperfect_info_stats) > 0:
        imperfect_ending_values = [getattr(imperfect_info_stats[i], value_name)[-1] for i in range(len(imperfect_info_stats))]
    plt.plot(varied_values, [np.linalg.norm(perfect_ending_values[i] - imperfect_ending_values[i]) for i in range(len(perfect_info_stats))])
    plt.xlim(min(varied_values), max(varied_values))
    plt.show()

def visualize_market(market: ContentMarket, stats: TestStats, agent_colors):
    G = nx.DiGraph()
    pos = {}
    for agent in market.agents:
        G.add_node(agent.index, color=agent_colors[agent.index])
        if isinstance(agent, Producer):
            pos[agent.index] = (agent.main_interest[0], 0)
        if isinstance(agent, Influencer):
            pos[agent.index] = (0, 1)
    G.add_node("External", color='white')
    pos["External"] = (1, 1)

    print(pos)
    
    for consumer in market.consumers:
        for agent in market.agents:
            if stats.consumer_stats[consumer.index].following_rates[-1][agent.index] > 0:
                G.add_edge(consumer.index, agent.index, weight=stats.consumer_stats[consumer.index].following_rates[-1][agent.index])
        if stats.consumer_stats[consumer.index].following_rates[-1][-1] > 0:
            G.add_edge(consumer.index, "External", weight=stats.consumer_stats[consumer.index].following_rates[-1][-1])
    
    for influencer in market.influencers:
        for agent in market.agents:
            if stats.influencer_stats[influencer.index].following_rates[-1][agent.index] > 0:
                G.add_edge(influencer.index, agent.index, weight=stats.influencer_stats[influencer.index].following_rates[-1][agent.index])

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u,v in edges]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    edge_weights = [(weight - min_weight) / (max_weight - min_weight) * 1 + 0.25 for weight in edge_weights]

    nx.draw(G, pos, with_labels=True, ax=ax, node_color=[G.nodes[n]['color'] for n in G.nodes], width=edge_weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #plt.axhline(-0.5, color='black', linewidth=0.5)  # Draws the x-axis
    plt.axis('on')
    ax.set_xlim(market.topics_bounds[0][0] - 0.25, market.topics_bounds[0][1] + 0.25)
    #plt.xticks(ticks=np.arange(-1, 1, 0.1), rotation=45)
    ax.tick_params(bottom=True, labelbottom=True)
    plt.show()

def visualize_influencer(market: ContentMarket, stats: InfluencerStats, agent_colors):
    G = nx.DiGraph()
    pos = {}
    for agent in market.agents:
        G.add_node(agent.index, color=agent_colors[agent.index])
        if isinstance(agent, Producer):
            pos[agent.index] = (agent.main_interest[0], 0)
        if isinstance(agent, Influencer):
            pos[agent.index] = (0, 1)
    
    for influencer in market.influencers:
        for agent in market.agents:
            if stats.following_rates[-1][agent.index] > 0:
                G.add_edge(influencer.index, agent.index, weight=stats.following_rates[-1][agent.index])

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u,v in edges]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    edge_weights = [(weight - min_weight) / (max_weight - min_weight) * 1 + 0.25 for weight in edge_weights]

    nx.draw(G, pos, with_labels=True, ax=ax, node_color=[G.nodes[n]['color'] for n in G.nodes], width=edge_weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #plt.axhline(-0.5, color='black', linewidth=0.5)  # Draws the x-axis
    plt.axis('on')
    ax.set_xlim(market.topics_bounds[0][0] - 0.25, market.topics_bounds[0][1] + 0.25)
    #plt.xticks(ticks=np.arange(-1, 1, 0.1), rotation=45)
    ax.tick_params(bottom=True, labelbottom=True)
    plt.show()

def visualize_influencer_followers(market: ContentMarket, stats: dict[int, ConsumerStats], agent_colors):
    G = nx.DiGraph()
    pos = {}
    for agent in market.agents:
        G.add_node(agent.index, color=agent_colors[agent.index])
        if isinstance(agent, Consumer):
            pos[agent.index] = (agent.main_interest[0], 0)
        if isinstance(agent, Influencer):
            pos[agent.index] = (0, 1)
    
    for influencer in market.influencers:
        for consumer in market.consumers:
            if stats[consumer.index].following_rates[-1][influencer.index] > 0:
                G.add_edge(consumer.index, influencer.index, weight=stats[consumer.index].following_rates[-1][influencer.index])

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u,v in edges]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    edge_weights = [(weight - min_weight) / (max_weight - min_weight) * 1 + 0.25 for weight in edge_weights]

    nx.draw(G, pos, with_labels=True, ax=ax, node_color=[G.nodes[n]['color'] for n in G.nodes], width=edge_weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #plt.axhline(-0.5, color='black', linewidth=0.5)  # Draws the x-axis
    plt.axis('on')
    ax.set_xlim(market.topics_bounds[0][0] - 0.25, market.topics_bounds[0][1] + 0.25)
    #plt.xticks(ticks=np.arange(-1, 1, 0.1), rotation=45)
    ax.tick_params(bottom=True, labelbottom=True)
    plt.show()

def visualize_consumer(market: ContentMarket, stats: ConsumerStats, agent_colors):
    G = nx.DiGraph()
    pos = {}
    for agent in market.agents:
        G.add_node(agent.index, color=agent_colors[agent.index])
        if agent == stats.agent:
            pos[agent.index] = (agent.main_interest[0], 0.5)
        elif isinstance(agent, Producer):
            pos[agent.index] = (agent.main_interest[0], 0)
        elif isinstance(agent, Influencer):
            pos[agent.index] = (1, 1)
    G.add_node("External", color='white')
    pos["External"] = (-1, 1)
    
    for agent in market.agents:
        if stats.following_rates[-1][agent.index] > 0:
            G.add_edge(stats.agent.index, agent.index, weight=stats.following_rates[-1][agent.index])
    G.add_edge(stats.agent.index, "External", weight=stats.following_rates[-1][-1])

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u,v in edges]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    edge_weights = [(weight - min_weight) / (max_weight - min_weight) * 1 + 0.25 for weight in edge_weights]

    nx.draw(G, pos, with_labels=True, ax=ax, node_color=[G.nodes[n]['color'] for n in G.nodes], width=edge_weights)
    
    plt.axis('on')
    ax.set_xlim(market.topics_bounds[0][0] - 0.25, market.topics_bounds[0][1] + 0.25)
    ax.tick_params(bottom=True, labelbottom=True)
    plt.show()