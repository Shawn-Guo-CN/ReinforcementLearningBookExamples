# ReinforcementLearningBookExamples
Example codes to implement the examples in Richard's book, 
Reinforcement Learning: An Introduction.

## Scripts
1. *1TenArmedBandits.py*: code scrip regarding to the k-armed bandit problem 
in Chapter 2.
    - [x] Bandit class
    - [x] Agent class
    - [x] epsilon-greedy algorithm
    - [x] optimistic initial values
    - [x] ucb
    - [x] gradient algorithm

2. *2GridWorld_Ch3.py*: code script regarding to example in Chapter 3
    - [x] GridWorld
    - [x] values estimated based on Bellman equation
    - [x] values estimated based on Bellman optimal equation
    
3. *3Carrental_Ch4.py*: code script for car rental problem in Chapter4.
However, there still exists bug in *maybe* **get_expected_return** function.
    - [x] JackRentalCompany, class simulating the car rental company
    - [x] *policy_evaluate* in Agent class
    - [x] *policy_improve* in Agent class
    - [ ] **debug!!!**
    
4. *13CliffWalking_Ch13.py*: code script for cliff walking problem in Chapter 13 with
various methods.
    - [ ] CliffWalk, class simulaitng the cliff walking game
    - [ ] REINFORCE
    - [ ] REINFORCE with Baseline
    - [ ] Actor-Critic w/o eligibility trace

## Todo:
1. Car Rental in Chapter 4
2. Various methods for solving Blackjack in Chapter 5
3. Q-learning for Cliff Walking in Chapter 6
4. REINFORCE with/without baseline for Corridor Grid World in Chapter 13
5. Actor-Critic in Chapte 13 (but, for what?)
