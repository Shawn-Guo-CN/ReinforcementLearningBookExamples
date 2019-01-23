#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/23 19:18
# @Author  : Shawn
# @File    : GridWorld.py


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


class GridWorld(object):
    def __init__(self, size=5, A_pos=[0,1], A_prime_pos=[4,1], B_pos=[0,3], B_prime_pos=[2,3]):
        self.size = size
        self.A_pos = np.asarray(A_pos)
        self.B_pos = np.asarray(B_pos)
        self.A_prime_pos = np.asarray(A_prime_pos)
        self.B_prime_pos = np.asarray(B_prime_pos)

        self.agent_location = np.zeros(2)

        self.action2shift = {
            'U': np.asarray([-1, 0]),
            'D': np.asarray([1, 0]),
            'L': np.asarray([0, -1]),
            'R': np.asarray([0, 1])
        }

    def set_agent_location(self, i, j):
        assert i < self.size and j < self.size
        self.agent_location = np.asarray([i, j])

    def get_agent_location(self):
        return self.agent_location

    def take_action(self, action):
        """
        :param action: str, 'U', 'D', 'L', 'R'
        """
        assert action in self.action2shift.keys()

        if np.array_equal(self.agent_location, self.A_pos):
            self.agent_location = self.A_prime_pos
            return 10
        elif np.array_equal(self.agent_location, self.B_pos):
            self.agent_location = self.B_prime_pos
            return 5
        elif (self.agent_location[0] == 0 and action == 'U') or \
                (self.agent_location[0] == self.size - 1 and action == 'D') or \
                (self.agent_location[1] == 0 and action == 'L') or \
                (self.agent_location[1] == self.size -1 and action == 'R'):
            # agent is currently at the top row and still wants to go up
            return -1
        else:
            self.agent_location += self.action2shift[action]
            return 0


# function for drawing values estimate
def draw_image(title, image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(image):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = 'white'

        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(range(len(image))):
        tb.add_cell(i, -1, width, height, text=label+1, loc='right',
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(range(len(image))):
        tb.add_cell(-1, j, width, height/2, text=label+1, loc='center',
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    plt.suptitle(title)
    plt.show()


def value_estimate_with_bellman_equation(gridworld, world_size=5, discount=0.9):
    values_est = np.zeros((world_size, world_size))
    while True:
        new_values_est = np.zeros((world_size, world_size))
        for i in range(0, world_size):
            for j in range(0, world_size):
                for action in ['U', 'D', 'L', 'R']:
                    gridworld.set_agent_location(i, j)
                    reward = gridworld.take_action(action)
                    new_pos = gridworld.get_agent_location()
                    new_values_est[i, j] += 0.25 * (reward + discount * values_est[new_pos[0], new_pos[1]])
        if np.sum(np.abs(values_est - new_values_est)) < 1e-4:
            break
        values_est = new_values_est
    draw_image('bellman equation', np.round(new_values_est, decimals=2))


def value_estimate_with_bellman_optimal_equation(gridworld, world_size=5, discount=0.9):
    values_est = np.zeros((world_size, world_size))
    while True:
        new_values_est = np.zeros((world_size, world_size))
        for i in range(0, world_size):
            for j in range(0, world_size):
                tmp_values_increment = []
                for action in ['U', 'D', 'L', 'R']:
                    gridworld.set_agent_location(i, j)
                    reward = gridworld.take_action(action)
                    new_pos = gridworld.get_agent_location()
                    tmp_values_increment.append(reward + discount * values_est[new_pos[0], new_pos[1]])
                new_values_est[i, j] = np.max(tmp_values_increment)
        if np.sum(np.abs(values_est - new_values_est)) < 1e-4:
            break
        values_est = new_values_est
    draw_image('bellman optimal equation', np.round(new_values_est, decimals=2))


if __name__ == '__main__':
    gridworld = GridWorld()
    value_estimate_with_bellman_equation(gridworld)
    value_estimate_with_bellman_optimal_equation(gridworld)
