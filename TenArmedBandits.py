#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/22 23:15
# @Author  : Shawn
# @File    : TenArmedBandits.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Bandit(object):
    def __init__(self, num_arms=10, min_mean=-2, max_mean=2, mu=1.):
        self.num_arms = num_arms
        self.means = np.random.uniform(min_mean, max_mean, self.num_arms)
        self.mus = [mu] * self.num_arms

    def get_reward_samples(self):
        random_samples = np.random.multivariate_normal(self.means, np.eye(self.num_arms), size=[200])
        return random_samples

    def take_arm(self, K):
        return np.random.normal(self.means[K], self.mus[K])


class Agent(object):
    def __init__(self):
        pass


def figure2_1(bandit):
    plt.figure(1)
    sns.violinplot(data=bandit.get_reward_samples())
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")


if __name__ == '__main__':
    bandit = Bandit()
    figure2_1(bandit)
    plt.show()
