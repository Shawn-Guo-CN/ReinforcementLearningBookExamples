import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

# configurations
gamma = 1.00
lr = 2e-12
best_score = -14
log_interval = 10
test_interval = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))
torch.manual_seed(2333)
np.random.seed(1234)


class CliffWalking(object):
    def __init__(self, step_limit=50):
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

        self.num_actions = len(self.actions)
        self.state_dim = 2

        # this is a rule added by Shawn
        self.steps = 0
        self.step_limit = step_limit

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
        if self.steps > self.step_limit:
            return self.pos, -100, True
        if isinstance(action, str):
            new_pos, reward, terminate = self.transmit_tensor[self.pos[0]][self.pos[1]][self.actions[action]]
        else:
            new_pos, reward, terminate = self.transmit_tensor[self.pos[0]][self.pos[1]][action]
        self.pos[0] = new_pos[0]
        self.pos[1] = new_pos[1]
        self.steps += 1
        return new_pos, reward, terminate

    def show_pos(self):
        env = np.zeros(self.shape)
        env[self.pos[0]][self.pos[1]] = 1
        print(env)

    def reset(self):
        self.pos = np.asarray([self.shape[0] - 1, 0])
        self.steps = 0
        return self.pos


class REINFORCE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=48):
        super(REINFORCE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        for m in self.modules():
            # in case add more modules later
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.fc2(x))
        return policy

    @classmethod
    def train_model(cls, model, transitions, optimizer, gamma=1.0):
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


def test_cliff_warlking_by_hand(cw):
    _, reward, terminate = cw.transmit_tensor[3][0][0]
    cw.show_pos()
    score = 0
    while not terminate:
        action = input('input your actoin (U/D/L/R):')
        new_pos, reward, terminate = cw.take_action(action)
        score += reward
        print(new_pos, reward, terminate, score)
        cw.show_pos()


def train_REINFORCE():
    cw = CliffWalking()
    input_dim = cw.state_dim
    output_dim = cw.num_actions
    print('state size:', input_dim)
    print('action size:', output_dim)

    model = REINFORCE(input_dim, output_dim)

    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    model.to(device)
    model.train()
    steps = 0

    for e in range(3000):
        score = 0
        state = cw.reset()
        state = torch.Tensor(state).to(device).unsqueeze(0)

        terminate = False
        replay_pool = ReplayPool()
        # create an episode
        while not terminate:
            steps += 1

            action = model.get_action(state)
            next_state, reward, terminate = cw.take_action(action)

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if terminate else 1
            reward = reward if not terminate or score == -14 else -1

            action_one_hot = torch.zeros(output_dim)
            action_one_hot[action] = 1
            replay_pool.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

        loss = REINFORCE.train_model(model, replay_pool.sample(), optimizer)
        print('[loss]episode %d: %.2f' % (e, loss))

        if e % test_interval == 0 and (not e == 0):
            returns = []
            model.eval()
            for i in range(100):
                terminate = False
                state = cw.reset()
                state = torch.Tensor(state).to(device)
                state = state.unsqueeze(0)
                total_return = 0
                while not terminate:
                    action = model.get_action(state)
                    next_state, reward, terminate = cw.take_action(action)
                    total_return += reward
                returns.append(total_return)
            model.train()
            print(e, np.mean(np.asarray(returns)))


if __name__ == '__main__':
    train_REINFORCE()
    # cw = CliffWalking()
    # test_cliff_warlking_by_hand(cw)
