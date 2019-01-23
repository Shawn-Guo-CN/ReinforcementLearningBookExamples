#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/22 23:15
# @Author  : Shawn
# @File    : TenArmedBandits.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Bandit(object):
    def __init__(self, num_arms=10, mean=0., mu=1.):
        self.num_arms = num_arms
        self.means = np.random.normal(mean, mu, self.num_arms)
        self.mus = [mu] * self.num_arms

        self.best_action = np.argmax(self.means)

    def get_reward_samples(self, dim=2000):
        # specifically for generating figure 2.1
        random_samples = np.random.multivariate_normal(self.means, np.eye(self.num_arms), size=[dim])
        return random_samples

    def take_arm(self, K):
        return np.random.normal(self.means[K], self.mus[K])


class Agent(object):
    def __init__(self, num_arm=10, epsilon=0., initial_value=0., step_size=0.1, sample_average=False,
                 UCB_param=None, gradient=False, gradient_baseline=False):
        self.num_arm = num_arm
        self.epsilon = epsilon

        self.initial_value = initial_value
        self.step_size = step_size
        self.sample_average = sample_average
        self.inidices = np.arange(self.num_arm)
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline

        self.average_reward = 0

        # number of taking every action
        self.action_counts = [0] * self.num_arm

        # index of current time step
        self.time_step = 0

        # estimated values for every arm
        self.values_est = [self.initial_value] * self.num_arm

    def reset(self):
        self.values_est = [self.initial_value] * self.num_arm
        self.action_counts = [0] * self.num_arm
        self.time_step = 0
        self.average_reward = 0

    def get_action(self):
        # probability with respect to explore
        if self.epsilon > 0:
            if np.random.binomial(1, self.epsilon) == 1:
                return np.random.choice(self.inidices)

        # probability with respect to exploit
        # if use UCB
        if self.UCB_param is not None:
            UCB_est = self.values_est + self.UCB_param * \
                      np.sqrt(np.log(self.time_step + 1) / (np.asarray(self.action_counts) + 1))
            return np.argmax(UCB_est)

        # if use gradient
        if self.gradient:
            exp_est = np.exp(self.values_est)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.inidices, p=self.action_prob)

        # otherwise
        return np.argmax(self.values_est)

    def update_values(self, K, reward):
        # get reward by taking K-th arm
        self.time_step += 1
        self.average_reward = (self.time_step - 1.0) / self.time_step * self.average_reward + \
                              reward / self.time_step
        self.action_counts[K] += 1

        if self.sample_average:
            self.values_est[K] += 1.0 / self.action_counts[K] * (reward - self.values_est[K])
        elif self.gradient:
            one_hot = np.zeros(self.num_arm)
            one_hot[K] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.values_est = self.values_est + self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            self.values_est[K] += self.step_size * (reward - self.values_est[K])


# game play function for all the image generating functions
def play_game(num_bandits, episode_length, agents, bandit_ini_mean=0.):
    best_action_counts = np.asarray([np.zeros(episode_length, dtype='float') for _ in range(len(agents))])
    rewards = np.asarray([np.zeros(episode_length, dtype='float') for _ in range(len(agents))])

    for i in range(num_bandits):
        bandit = Bandit(mean=bandit_ini_mean)
        for agent_idx, agent in enumerate(agents):
            # remember resetting the status of agent when it comes into a new episode
            agent.reset()
            for t in range(episode_length):
                action = agent.get_action()
                reward = bandit.take_arm(action)
                agent.update_values(action, reward)
                rewards[agent_idx][t] += reward
                if action == bandit.best_action:
                    best_action_counts[agent_idx][t] += 1

    return best_action_counts / num_bandits, rewards / num_bandits


figure_index = 0


# function for visualising k-armed bandit rewards
def figure2_1(bandit, dim=2000):
    global figure_index
    plt.figure(figure_index)
    figure_index += 1
    sns.violinplot(data=bandit.get_reward_samples(dim=dim))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")


# generate figure 2.2
def epsilon_greedy(num_bandits, episode_length):
    epsilons = [0., 0.1, 0.01]
    agents = []
    for ep_idx, ep in enumerate(epsilons):
        agents.append(Agent(epsilon=ep, sample_average=True))
    best_action_counts, average_rewards = play_game(num_bandits, episode_length, agents)
    global figure_index
    plt.figure(figure_index)
    figure_index += 1
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = ' + str(eps))
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.figure(figure_index)
    figure_index += 1
    for eps, rewards in zip(epsilons, average_rewards):
        plt.plot(rewards, label='epsilon = ' + str(eps))
    plt.xlabel('Steps')
    plt.ylabel('average reward')
    plt.legend()


# generate figure 2.3
def optimal_initial_values(num_bandits, episode_length):
    agents = [Agent(epsilon=0, initial_value=5, step_size=.1),
              Agent(epsilon=0.1, initial_value=0, step_size=0.1)]
    best_action_counts, _ = play_game(num_bandits, episode_length, agents)
    global figure_index
    plt.figure(figure_index)
    plt.plot(best_action_counts[0], label='epsilon = 0, q = 5')
    plt.plot(best_action_counts[1], label='epsilon = 0.1, q = 0')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()


# for figure 2.4
def ucb(num_bandits, episode_length):
    agents = [Agent(epsilon=0, step_size=0.1, UCB_param=2),
              Agent(epsilon=0.1, step_size=0.1)]
    _, avg_rewards = play_game(num_bandits, episode_length, agents)
    global figure_index
    plt.figure(figure_index)
    figure_index += 1
    plt.plot(avg_rewards[0], label='UCB c = 2')
    plt.plot(avg_rewards[1], label='epsilon greedy epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()


# for figure 2.5
def gradient_algorithm(num_bandits, episode_length):
    agents = [Agent(gradient=True, step_size=0.1, gradient_baseline=True),
              Agent(gradient=True, step_size=0.1, gradient_baseline=False),
              Agent(gradient=True, step_size=0.4, gradient_baseline=True),
              Agent(gradient=True, step_size=0.4, gradient_baseline=False)]

    best_action_counts, _ = play_game(num_bandits, episode_length, agents, 4.)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']
    global figure_index
    plt.figure(figure_index)
    figure_index += 1
    for i in range(0, len(agents)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()


# following three example functions are for understanding the playing procedure
# following is for testing whether agent and bandit are rightly implemented
def play_1bandit_1agent():
    bandit = Bandit()
    agent = Agent(epsilon=0.05, sample_average=True)

    total_reward = 0.
    right_times = 0

    avg_rewards = []
    right_times_freq = []

    for t in range(1, 1001):
        action = agent.get_action()
        reward = bandit.take_arm(action)
        agent.update_values(action, reward)
        total_reward += reward
        if action == bandit.best_action:
            right_times += 1
        avg_rewards.append(total_reward / t)
        right_times_freq.append(right_times / t)

    plt.plot(right_times_freq, label='right freq')
    plt.plot(avg_rewards, label='avg reward')
    plt.plot([bandit.means[bandit.best_action]] * 1000, label='best reward')
    plt.legend()


# following is for testing whether agent and bandit are rightly implemented
def play_1bandit_3agents():
    bandit = Bandit()

    epsilons = [0., 0.1, 0.01]
    agents = []
    for ep_idx, ep in enumerate(epsilons):
        agents.append(Agent(epsilon=ep, sample_average=True))

    total_reward = [0.] * 3
    right_times = [0] * 3

    avg_rewards = [[], [], []]
    right_times_freq = [[], [], []]

    for agent_idx, agent in enumerate(agents):
        for t in range(1, 1001):
            action = agent.get_action()
            reward = bandit.take_arm(action)
            agent.update_values(action, reward)
            total_reward[agent_idx] += reward
            if action == bandit.best_action:
                right_times[agent_idx] += 1
            avg_rewards[agent_idx].append(total_reward[agent_idx] / t)
            right_times_freq[agent_idx].append(right_times[agent_idx] / t)

    plt.figure(1)
    for i in range(3):
        plt.plot(right_times_freq[i], label='right freq ' + str(epsilons[i]))
    plt.legend()

    plt.figure(2)
    plt.plot([bandit.means[bandit.best_action]] * 1000, label='best reward')
    for i in range(3):
        plt.plot(avg_rewards[i], label='avg reward ' + str(epsilons[i]))
    plt.legend()


def play_nbandit_1agent(num_bandit, time_step):
    agent = Agent(epsilon=0.1, sample_average=True)

    bandits = [Bandit() for _ in range(num_bandit)]

    rewards = np.zeros(time_step, dtype='float')
    best_action_counts = np.zeros(time_step)

    for bandit_idx, bandit in enumerate(bandits):
        agent.reset()
        for t in range(time_step):
            action = agent.get_action()
            reward = bandit.take_arm(action)
            agent.update_values(action, reward)
            rewards[t] += reward
            if action == bandit.best_action:
                best_action_counts[t] += 1

    plt.figure(1)
    plt.plot(rewards / num_bandit, label='rewards')
    plt.plot(best_action_counts / num_bandit, label='best action counts')
    plt.legend()


if __name__ == '__main__':
    # epsilon_greedy(2000, 1000)
    # optimal_initial_values(2000, 1000)
    # ucb(2000, 1000)
    gradient_algorithm(2000, 1000)
    plt.show()
