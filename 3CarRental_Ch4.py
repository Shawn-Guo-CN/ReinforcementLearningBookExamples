#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/23 20:58
# @Author  : Shawn
# @File    : 3CarRental_Ch4.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson
from math import *

THETA = 100


class JackRentalCompany(object):
    def __init__(self, max_capacity=20, max_move=5, lambdas=[3, 4, 3, 2], rental_credit=10, move_cost=2):
        self.max_capacity = max_capacity
        self.max_move = max_move

        self.rental_credit = 10.
        self.move_cost = 2.

        self.lambda_request_1st = lambdas[0]
        self.lambda_request_2nd = lambdas[1]
        self.lambda_return_1st = lambdas[2]
        self.lambda_return_2nd = lambdas[3]

    def get_expected_return(self, state, action, values_est, gamma=0.9, approximate_return=True):
        G = 0.0
        G -= self.move_cost * abs(action)

        # execute action
        num_cars_1st = int(min(state[0] - action, self.max_capacity))
        num_cars_2nd = int(min(state[1] + action, self.max_capacity))

        for rental_request_1st in range(num_cars_1st + 1):
            for rental_request_2nd in range(num_cars_2nd + 1):
                # here, we assume all rental requests are given before any return
                num_rental_1st = min(num_cars_1st, rental_request_1st)
                num_rental_2nd = min(num_cars_2nd, rental_request_2nd)

                # rent cars to customers
                reward = (num_rental_1st + num_rental_2nd) * self.rental_credit
                num_cars_1st -= num_rental_1st
                num_cars_2nd -= num_rental_2nd

                prob = poisson.pmf(num_rental_1st, self.lambda_request_1st) * \
                       poisson.pmf(num_rental_2nd, self.lambda_request_2nd)

                if approximate_return:
                    num_cars_return_1st = self.lambda_return_1st
                    num_cars_return_2nd = self.lambda_return_2nd
                    num_cars_1st = min(num_cars_1st + num_cars_return_1st, self.max_capacity)
                    num_cars_2nd = min(num_cars_2nd + num_cars_return_2nd, self.max_capacity)
                    G += prob * (reward + gamma * values_est[num_cars_1st, num_cars_2nd])

        return G


class Agent(object):
    def __init__(self, max_capacity=20, max_move=5, gamma=0.9):
        self.state_space_dim = max_capacity + 1
        self.action_space = np.arange(-max_move, max_move + 1)
        self.gamma = gamma

        self.states = []
        for i in range(self.state_space_dim):
            for j in range(self.state_space_dim):
                self.states.append([i, j])

        self.policy = np.zeros((self.state_space_dim, self.state_space_dim), dtype='int')
        self.values_est = np.zeros((self.state_space_dim, self.state_space_dim), dtype='float')

    def policy_evaluate(self, company):
        policy_converged = False

        old_values_est = self.values_est.copy()

        for state in self.states:
            self.values_est[state[0], state[1]] = \
                company.get_expected_return(state, self.policy[state[0], state[1]], old_values_est, self.gamma)

        error_sum = np.sum(np.abs(old_values_est - self.values_est))
        print('\t\terror sum:', error_sum)
        if error_sum < THETA:
            policy_converged = True
        del old_values_est

        return policy_converged

    def policy_improve(self, company):
        policy_stable = True
        num_policy_changes = 0

        for state in self.states:
            old_action = self.policy[state[0], state[1]]
            action_returns = []
            for action in self.action_space:
                if (0 <= action <= state[0]) or (action < 0 and state[1] >= -action):
                    action_returns.append(company.get_expected_return(state, action, self.values_est, self.gamma))
                else:
                    action_returns.append(-float('inf'))
            self.policy[state[0], state[1]] = self.action_space[np.argmax(action_returns)]
            if not old_action == self.policy[state[0], state[1]]:
                policy_stable = False
                num_policy_changes += 1

        return policy_stable, num_policy_changes


def policy_iterate(company, agent):
    policy_improvement_converge_flag = False
    eva_iter_num = 0
    improve_iter_num = 0
    while not policy_improvement_converge_flag:
        policy_evaluation_converge_flag = False
        while not policy_evaluation_converge_flag:
            eva_iter_num += 1
            print('[Evaluation]doing %d-th time policy evaluation' % eva_iter_num)
            policy_evaluation_converge_flag = agent.policy_evaluate(company)
        eva_iter_num = 0

        improve_iter_num += 1
        print('[Improvement]doing %d-th time policy improvement' % improve_iter_num)
        policy_improvement_converge_flag, policy_update_num = agent.policy_improve(company)
        print('[Result]%d states has been updated in policy improvement' % policy_update_num)
        print('-------------------------------------------------------------------------')


# plot a policy/state value matrix
def draw_matrix(data, labels, dim=21):
    AxisXPrint = []
    AxisYPrint = []
    pos = []
    for i in range(0, dim):
        for j in range(0, dim):
            AxisXPrint.append(i)
            AxisYPrint.append(j)
            pos.append([i, j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    AxisZ = []
    for i, j in pos:
        AxisZ.append(data[i, j])
    ax.scatter(AxisXPrint, AxisYPrint, AxisZ)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])


if __name__ == '__main__':
    company = JackRentalCompany()
    agent = Agent()
    policy_iterate(company, agent)
    draw_matrix(agent.policy,
         ['# of cars in first location', '# of cars in second location', '# of cars to move during night'])
    draw_matrix(agent.values_est,
         ['# of cars in first location', '# of cars in second location', 'expected returns'])
    plt.show()
