#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:01:21 2019

@author: xiayezi
"""

import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))
log_interval = 5

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


torch.manual_seed(1234)
np.random.seed(1234)

#=========================== Environment Class=================================
#1. Class:: CliffWalking: Fundamental environment
#2. Function:: test_cliff_warlking_by_hand: use keyboard to test the environment
#3. Class:: ReplayPool: use a bidirection queue to store the state history
#
#==============================================================================


class CliffWalking(object):
    def __init__(self, birth=[3,0]):
        self.len_state_space = 4*12
        self.shape = (4, 12)
        self.birth = birth
        self.pos = tuple(birth)
        self.cliff = np.zeros(self.shape, dtype=np.bool)
        self.cliff[-1, 1:-1] = 1
        self.done = False  # Show whether the game is over
        self.num_action = 4
        self.actions = {
            'U': 0,
            'D': 1,
            'L': 2,
            'R': 3
        }

        self.action2shift = {
            0: [-1, 0],
            1: [1, 0],
            2: [0, -1],
            3: [0, 1]
        }

    def take_action(self, action):
        self.done = False
        reward = -1
        pos_0,pos_1 = self.pos[0],self.pos[1]
        if isinstance(action,str):
            action = self.actions[action]
        pos_0 += self.action2shift[action][0]
        pos_1 += self.action2shift[action][1]
        if pos_0<0 or pos_0>self.shape[0]-1 or pos_1<0 or pos_1>self.shape[1]-1:
            reward = -1
            pos_0 = max(pos_0, 0)
            pos_0 = min(pos_0, self.shape[0] - 1)
            pos_1 = max(pos_1, 0)
            pos_1 = min(pos_1, self.shape[1] - 1)
        
        self.pos = (pos_0,pos_1)
        
        if self.cliff[self.pos[0]][self.pos[1]]:
            reward = -100.
            self.reset()
            self.done = False
            
        elif self.pos[0] == self.shape[0] - 1 and self.pos[1] == self.shape[1] - 1:
            self.done = True
            reward = 0
        else: pass

        return self.pos, reward, self.done

    def show_pos(self):
        env = np.zeros(self.shape)
        env[self.pos[0]][self.pos[1]] = 1
        print(env)
    
    def show_path(self,states):
        env = np.zeros(self.shape)
        cnt = 0
        for s in states:
            cnt += 1
            env[s[0]][s[1]]=cnt
        print(env)
    
    def get_pos(self):
        return self.pos
    
    def reset(self):
        self.pos = tuple(self.birth)
        self.done = False
        return self.pos, self.done
    
def test_cliff_warlking_by_hand(env):
    _,terminate = env.reset()
    env.show_pos()
    score = 0
    while not terminate:
        action = input('input your actoin (U/D/L/R):')
        new_pos, reward, terminate = env.take_action(action)
        score += reward
        print(new_pos, reward, terminate, score)
        env.show_pos()    

class ReplayPool(object):
    def __init__(self):
        self.memory = deque()

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def pop_all(self):
        memory = self.memory
        return Transition(*zip(*memory))

    def reset(self):
        self.memory = deque()

    def __len__(self):
        return len(self.memory)

#============================Assistant functions===============================
#
#
#
#==============================================================================


 
def epsilon_greedy_policy(Q,state,num_action,epsilon=0.1,greedy=False):
    '''
    greedy_action = np.argmax(Q[state])
    if np.random.uniform(0,1)>epsilon or greedy == True:      # Greedy
        return greedy_action
    else:
        return np.random.randint(0,num_action-1)
    '''
    best_action = np.argmax(Q[state])
    A =np.ones(num_action,dtype=np.float32)*epsilon/num_action
    A[best_action] += 1-epsilon
    if greedy==False:
        return np.random.choice(np.arange(num_action),p=A)
    else:
        return best_action

def test_cliff(Q,max_step=300):
    steps = 0
    env_test = CliffWalking()
    state, terminate = env_test.reset()
    action_store = []
    state_store = []
    sum_reward = 0
    while not terminate:
        steps += 1
        action = epsilon_greedy_policy(Q,state,env.num_action,greedy=False)
        state, reward, terminate = env_test.take_action(action)
        sum_reward += reward
        action_store.append(action)
        state_store.append(state)
        if steps > max_step:
            print('Out of steps')
            return state_store, sum_reward, action_store
    return state_store, sum_reward, action_store

def state_to_onehot(state,len_state_space):
   '''
   Convert a state input, e.g. tuple(x,y) to a one-hot tensor
   '''
   state_one_hot = np.zeros(len_state_space)
   state_one_hot[state[0]*4 + state[1]] = 1
   state_one_hot = torch.Tensor(state_one_hot).unsqueeze(0)
   return state_one_hot

#============================Learning Functions================================
#1. Function:: sarsa: fundamental sarsa implementation [O]
#2. Class:: REINFORCE: REINFORCE algorithm, MC+PG, use an NN to approximate 
#           the policy function.
#
#==============================================================================          

def sarsa(env,episode_nums,discount_factor=1.0, alpha=0.5,Q=None):
    '''
        How to use:
            env = CliffWalking()
            episode_nums=10000
            Q,rewards = sarsa(env,episode_nums)

            states,reward,actions = test_cliff(Q)
            env.show_path(states)
    '''
    #env = CliffWalking()
    if Q==None:
        Q = defaultdict(lambda:np.zeros(env.num_action))
    rewards=[]
    epsilon = 0.1
    for i_episode in range(1,1+episode_nums):
        state, terminate = env.reset()
        action = epsilon_greedy_policy(Q,state,env.num_action,epsilon=epsilon,greedy=False)
        sum_reward = 0
        while not terminate:
            new_state,reward,terminate = env.take_action(action)
            if terminate:
                Q[state][action]=Q[state][action]+alpha*(reward+discount_factor \
                 *0.0-Q[state][action])
                break
            else:
                new_action =  epsilon_greedy_policy(Q,new_state,env.num_action,
                                                    epsilon=epsilon,greedy=False)
                Q[state][action]=Q[state][action]+alpha*(reward+discount_factor*
                 Q[new_state][new_action]-Q[state][action])
                state = new_state
                action = new_action
            sum_reward += reward
        rewards.append(sum_reward)
        
        if i_episode % 100 == 5:
            epsilon = epsilon*0.9
            print("\rEpisode {}/{}. epsilon={}".format(i_episode, episode_nums,epsilon))
            sys.stdout.flush()
            states,test_reward,actions = test_cliff(Q)
            if test_reward < -300:
                print(states)
                print(Q[states[-1]])
            rewards.append(test_reward)
            
    return Q,rewards


class AC_Policy(nn.Module):
    '''
        The ploicy network, only contain forward function
    '''
    def __init__(self, input_dim, output_dim, hidden_dim=50):
        super(AC_Policy, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim    # Params. of NN
        self.save_actions = []
        self.save_rewards = []

        self.affine_1 = nn.Linear(input_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim,1)
    
    def forward(self, x):
        '''
            Direct input state, one-hot should be done in the function
        '''
        x = F.relu(self.affine_1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values



def AC_select_action(state,greedy=False):
    '''
        Input: state: (x,y)
               greedy: T/F
        Output: action: number from 0 to 3
        State is the input of the policy network, and we can get a distribution
        of differnet actions. Sample the action according to the probabilty or 
        just use greedy method (controled by input flag-greedy)
    '''
    #state_tensor = torch.Tensor(state).unsqueeze(0)
    #probs = policy(state_tensor)
    
    state_onehot = state_to_onehot(state, env.len_state_space)
    probs, state_value = policy(state_onehot)    
    m = Categorical(probs)
    if greedy == False:
        action = m.sample()     # Sample according to the policy distribution
    else:
        action = torch.argmax(probs) # Select the best action according to prob.
    policy.save_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()



def finish_episode(gamma=1):
    '''
        When one episode finished, count the rewards and actions in the path.
    '''
    R = 0
    save_actions = policy.save_actions
    policy_loss = []
    value_loss = []
    vt = []
    for r in policy.save_rewards[::-1]:      # The rewards for each step is stored
        R = r + gamma*R
        vt.insert(0,R)             # list.insert means add item at specific posistion
    vt = torch.FloatTensor(vt)     # This vt is the estimated rewards along the path 
    vt = (vt-vt.mean())/(vt.std()+eps)  # Normalization of rewards
    for (log_prob, value), r in zip(save_actions, vt):
        reward = r-value.item()
        policy_loss.append(-log_prob*reward)        # Gradient update function
        value_loss.append(F.smooth_l1_loss(value,torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()+torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del policy.save_rewards[:]
    del policy.save_actions[:]


def AC_train(episode_nums=1000):
    
    for i_episode in range(episode_nums):
        state, terminate = env.reset()
        for t in range(20000):
            action = AC_select_action(state,greedy=False)
            state, reward, terminate = env.take_action(action)
            policy.save_rewards.append(reward)   # Store the reward each step
            if terminate:
                break
        episode_reward = np.array(policy.save_rewards).sum()   # Reward for this episode is the sum of all steps
        finish_episode()        # Update the parameters    

        if i_episode % log_interval == 0:
            print('Episode {}\t\t Episode rewards: {}'.format(
                i_episode, episode_reward))
    return episode_reward








env = CliffWalking()
episode_nums=5000
policy = AC_Policy(env.len_state_space,4)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
eps = np.finfo(np.float32).eps.item()

final_reward = AC_train(episode_nums)


#Q,rewards = sarsa(env,episode_nums)
#states,reward,actions = test_cliff(Q)
#env.show_path(states)












