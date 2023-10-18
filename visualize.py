import matplotlib.pyplot as plt
import numpy as np

from agents import Consumer, Producer, Agent, Influencer

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

def plot_topic_distribution_histogram(title, agents: list[Consumer | Producer], min_topic, max_topic, bins=50):
    plt.figure()
    plt.title(title)
    plt.xlabel('Topic')
    plt.ylabel('Frequency')
    plt.hist([agent.main_interest[0] for agent in agents], bins=bins)
    plt.yticks(range(0, len(agents) + 1))
    plt.xticks(range(min_topic, max_topic + 1))
    plt.show()

def plot_consumer_topic_interest_distributions(title, agents: list[Consumer], min_topic, max_topic, agent_colors):
    plt.figure()
    plt.title(title)
    topics_dist = np.linspace(min_topic, max_topic, 100)
    for a in agents:
        plt.plot(topics_dist, [a.consumption_topic_interest(np.array(t).reshape((1,))) for t in topics_dist], color=agent_colors[a.index], label=f"Consumer {a.index}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(0, 1.1)
    plt.show()

def plot_producer_topic_probability_distributions(title, agents: list[Producer], min_topic, max_topic, agent_colors):
    plt.figure()
    plt.title(title)
    topics_dist = np.linspace(min_topic, max_topic, 100)
    for a in agents:
        plt.plot(topics_dist, [a.topic_probability(np.array(t).reshape((1,))) for t in topics_dist], color=agent_colors[a.index], label=f"Producer {a.index}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(0, 1.1)
    plt.show()

def plot_agent_utility_by_iteration(title, agents: list[Agent], agent_colors, agent_stats, averages=None):
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for agent in agents:
        plt.plot(agent_stats[agent.index]['utilities'], label=get_agent_title(agent), color=agent_colors[agent.index])
    plt.legend()
    plt.xticks(range(0, len(agent_stats[agents[0].index]['utilities'])))
    plt.show()

def plot_agent_attention_used_by_iteration(title, agents: list[Consumer | Influencer], agent_colors, agent_stats, averages=None):
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for agent in agents:
        plt.plot(list(range(0, len(agent_stats[agent.index]['attention_used']))), agent_stats[agent.index]['attention_used'], label=get_agent_title(agent), color=agent_colors[agent.index])
    plt.legend()
    plt.xticks(range(0, len(agent_stats[agents[0].index]['attention_used'])))
    plt.ylim(0, max(agent.attention_bound for agent in agents) + 1)
    plt.show()

def plot_producer_topic_produced_by_iteration(title, producers: list[Producer], consumers: list[Consumer], agent_colors, agent_stats):
    plt.figure()
    plt.title(title)
    for producer in producers:
        plt.plot(agent_stats[producer.index]['topics'], label='Producer {}'.format(producer.index), color=agent_colors[producer.index])
    for consumer in consumers:
        plt.plot([consumer.main_interest] * len(agent_stats[producer.index]['topics']), label='Consumer {}'.format(consumer.index), linestyle='--', color=agent_colors[consumer.index])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(min(min(agent.main_interest for agent in producers), min(agent.main_interest for agent in consumers)) - 1, max(max(agent.main_interest for agent in producers), max(agent.main_interest for agent in consumers)) + 1)
    plt.xticks(range(len(agent_stats[producers[0].index]['topics'])))
    plt.show()

def plot_agent_following_rate_change_by_iteration(title, agents: list[Consumer | Influencer], agent_colors, agent_stats, averages=None):
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for agent in agents:
        plt.plot(agent_stats[agent.index]['rate_change'], label=get_agent_title(agent), color=agent_colors[agent.index])
    plt.legend()
    plt.xticks(range(0, len(agent_stats[agents[0].index]['rate_change'])))
    plt.show()

def plot_producer_topic_change_by_iteration(title, producers: list[Producer], agent_colors, agent_stats, averages=None):
    plt.figure()
    plt.title(title)
    if averages:
        plt.plot(averages, label="Average")
    for producer in producers:
        plt.plot(agent_stats[producer.index]['topic_change'], label='Producer {}'.format(producer.index), color=agent_colors[producer.index])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(len(agent_stats[producers[0].index]['topic_change'])))
    plt.show()

def plot_total_social_welfare_by_iteration(title, totals):
    plt.figure()
    plt.title(title)
    plt.plot(totals)
    plt.xticks(range(0, len(totals)))
    plt.show()

def plot_following_rate_by_main_interest_closeness(title, consumers: list[Consumer], producers: list[Producer], agent_colors, agent_stats):
    plt.figure()
    plt.title(title)
    plt.xlabel('Main interest closeness')
    plt.ylabel('Following rate')
    for consumer in consumers:
        interest_closeness = []
        following_rate = []
        for producer in producers:
            ending_rate = agent_stats[consumer.index]['following_rates'][-1][producer.index]
            interest_closeness.append(np.linalg.norm(consumer.main_interest - producer.main_interest))
            following_rate.append(ending_rate)
        plt.plot(interest_closeness, following_rate, label='Consumer {}'.format(consumer.index), color=agent_colors[consumer.index], marker='o')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plot_following_rates_by_iteration(agents: list[Consumer | Influencer], follows: list[Producer | Influencer], agent_colors, agent_stats):
    fig = plt.figure(figsize=(5, 20))
    for i, agent in enumerate(agents):
        ax = fig.add_subplot(len(agents), 1, i + 1)
        ax.set_title(f"Following rates by Iteration for {get_agent_title(agent)}")
        for other in follows:
            rate_by_iteration = [vec[other.index] for vec in agent_stats[agent.index]['following_rates']]
            ax.plot(rate_by_iteration, label=get_agent_title(other), color=agent_colors[other.index])
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_xticks(range(len(rate_by_iteration)))
        ax.sharex(fig.axes[0])
        ax.label_outer()
    plt.show()