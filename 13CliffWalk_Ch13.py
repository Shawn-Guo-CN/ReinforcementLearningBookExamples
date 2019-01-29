import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from tensorboardX import SummaryWriter
from collections import namedtuple, deque

# configurations
env_name = 'CartPole-v1'
gamma = 0.99
lr = 0.001
goal_score = 200
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class ReplayPool(object):
    def __init__(self):
        self.memory = deque()

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def sample(self):
        memory = self.memory
        return Transition(*zip(*memory))

    def __len__(self):
        return len(self.memory)


class CliffWalking(object):
    def __init__(self):
        self.shape = (4, 12)

        # always start from the left-dow corner
        self.pos = np.asarray([self.shape[0] - 1, 0])

        # build a
        self.cliff = np.zeros(self.shape, dtype=np.bool)
        self.cliff[-1, 1:-1] = 1

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

        self.transmit_tensor = self._build_transmit_tensor_()

    def _build_transmit_tensor_(self):
        trans_matrix = [[[] for _ in range(self.shape[1])] for __ in range(self.shape[0])]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for a in range(len(self.actions)):
                    trans_matrix[i][j].append(self._cal_new_position_((i, j), a))

        return trans_matrix

    def _cal_new_position_(self, old_pos, action):
        old_pos = np.asarray(old_pos)
        new_pos = old_pos + self.action2shift[action]
        new_pos[0] = max(new_pos[0], 0)
        new_pos[0] = min(new_pos[0], self.shape[0] - 1)
        new_pos[1] = max(new_pos[1], 0)
        new_pos[1] = min(new_pos[1], self.shape[1] - 1)

        reward = -1.
        terminate = False

        if self.cliff[old_pos[0]][old_pos[1]]:
            reward = -100.
            new_pos[0] = self.shape[0] - 1
            new_pos[1] = 0
            terminate = True

        if old_pos[0] == self.shape[0] - 1 and old_pos[1] == self.shape[1] - 1:
            terminate = True

        return new_pos, reward, terminate

    def take_action(self, action):
        new_pos, reward, terminate = self.transmit_tensor[self.pos[0]][self.pos[1]][self.actions[action]]
        self.pos[0] = new_pos[0]
        self.pos[1] = new_pos[1]
        return new_pos, reward, terminate

    def show_pos(self):
        env = np.zeros(self.shape)
        env[self.pos[0]][self.pos[1]] = 1
        print(env)

    def reset(self):
        self.pos = np.asarray([self.shape[0] - 1, 0])
        return self.pos


class REINFORCE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(REINFORCE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

        for m in self.modules():
            # in case add more modules later
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.fc2(x))
        return policy

    @classmethod
    def train_model(cls, model, transitions, optimizer):
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)

        returns = torch.zeros_like(rewards)

        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return

        policies = model(states)
        policies = policies.view(-1, model.output_dim)

        log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)

        loss = (-log_policies * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        policy = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.output_dim, 1, p=policy)[0]
        return action


def test_cliff_warlking_by_hand(cw):
    _, reward, terminate = cw.transmit_tensor[3][0][0]
    cw.show_pos()
    while not terminate:
        action = input('input your actoin (U/D/L/R):')
        new_pos, reward, terminate = cw.take_action(action)
        print(new_pos, reward, terminate)
        cw.show_pos()


def tmp_use_gym():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = REINFORCE(num_inputs, num_actions)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    net.to(device)
    net.train()
    running_score = 0
    steps = 0
    loss = 0

    for e in range(3000):
        done = False
        replay_pool = ReplayPool()

        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            action = net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            action_one_hot = torch.zeros(2)
            action_one_hot[action] = 1
            replay_pool.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

        loss = REINFORCE.train_model(net, replay_pool.sample(), optimizer)

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f}'.format(
                e, running_score))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            break


if __name__ == '__main__':
    cw = CliffWalking()
    tmp_use_gym()
