#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/23 20:58
# @Author  : Shawn
# @File    : 3CarRental_Ch4.py

import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.stats import poisson


theta = 1e-4


class JackRentalCompany(object):
    def __init__(self, max_capacity=20, max_move=5, lambdas=[3,4,3,2], rental_credit=10, move_cost=2):
        self.max_capacity = max_capacity
        self.max_move = max_move

        self.rental_credit = 10
        self.move_cost = 2

        self.lambda_request_1st = lambdas[0]
        self.lambda_request_2nd = lambdas[1]
        self.lambda_return_1st = lambdas[2]
        self.lambda_return_2nd = lambdas[3]


    def get_expected_return(self, state, num_cars_move, values_est):
        G = 0.0
        G -= self.move_cost * num_cars_move

        for request_1st in range(0, state[0]):
            for request_2nd in range(0, state[1]):
                num_cars_1st = int(min(state[0] - num_cars_move, self.max_capacity))
                num_cars_2nd = int(min(state[1] + num_cars_move, self.max_capacity))

                valid_request_1st = min(num_cars_1st, request_1st)
                valid_request_2nd = min(num_cars_2nd, request_2nd)

                reward = self.rental_credit * (valid_request_1st + valid_request_2nd)
                num_cars_1st -= valid_request_1st
                num_cars_2nd -= valid_request_2nd

                prob = poisson.pmf(request_1st, self.lambda_request_1st) * \
                    poisson.pmf(request_2nd, self.lambda_request_2nd)
