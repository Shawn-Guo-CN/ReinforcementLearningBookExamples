import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

# configurations
gamma = 1.00
lr = 1e-3
best_score = -14
log_interval = 10
test_interval = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))
torch.manual_seed(1234)
np.random.seed(1234)


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

        self.num_actions = len(self.actions)
        self.state_dim = 2

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
            terminate = False

        if old_pos[0] == self.shape[0] - 1 and old_pos[1] == self.shape[1] - 1:
            terminate = True
            reward = 0.
            new_pos[0] = self.shape[0] - 1
            new_pos[1] = self.shape[1] - 1

        return new_pos, reward, terminate

    def take_action(self, action):
        if isinstance(action, str):
            new_pos, reward, terminate = self.transmit_tensor[self.pos[0]][self.pos[1]][self.actions[action]]
        else:
            new_pos, reward, terminate = self.transmit_tensor[self.pos[0]][self.pos[1]][action]
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
    def __init__(self, input_dim, output_dim, hidden_dim=96):
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

        optimizer.zero_grad()
        loss = (-log_policies * returns).sum()

        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state, test=False):
        policy = self.forward(state)
        policy = policy[0].data.numpy()

        if test:
            action = np.argmax(policy)
        else:
            action = np.random.choice(self.output_dim, 1, p=policy)[0]
        return action


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=48):
        super(ActorCritic, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            # in case add more modules later
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.selu(self.fc_hidden(x))
        policy = F.softmax(self.fc_actor(x))
        q_value = self.fc_critic(x)
        return policy, q_value

    @classmethod
    def train_model(cls, model, transition, optimizer, gamma=1.0):
        state, next_state, action, reward, mask = transition

        policy, q_value = model(state)
        policy, q_value = policy.view(-1, model.output_dim), q_value.view(-1, 1)
        _, next_q_value = model(next_state)
        next_q_value = next_q_value.view(-1, 1)
        next_action = model.get_action(next_state)

        target = reward + mask * gamma * next_q_value[0]

        log_policy = torch.log(policy[0])[action]
        loss_policy = -log_policy * q_value[0]
        loss_value = F.mse_loss(q_value[0], target.detach())

        loss = loss_policy + loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state, test=False):
        policy, _ = self.forward(state)
        policy = policy[0].data.numpy()

        if test:
            action = np.argmax(policy)
        else:
            action = np.random.choice(self.output_dim, 1, p=policy)[0]
        return action


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


def convert_state2onehot(state):
    state_one_hot = np.zeros(48)
    state_one_hot[state[0] * 12 + state[1]] = 1.
    state_one_hot = torch.Tensor(state_one_hot).to(device).unsqueeze(0)
    return state_one_hot


def train_REINFORCE(env):
    model = REINFORCE(48, env.num_actions)

    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    model.to(device)
    model.train()
    replay_pool = ReplayPool()

    for e in range(3000):
        state = env.reset()
        terminate = False
        replay_pool.reset()

        # create an episode
        while not terminate:
            state_one_hot = convert_state2onehot(state)

            action = model.get_action(state_one_hot)
            next_state, reward, terminate = env.take_action(action)

            mask = 0 if terminate else 1

            next_state_one_hot = convert_state2onehot(next_state)

            action_one_hot = torch.zeros(output_dim)
            action_one_hot[action] = 1
            replay_pool.push(state_one_hot, next_state_one_hot, action_one_hot, reward, mask)

            state = next_state

        loss = model.train_model(model, replay_pool.pop_all(), optimizer)

        print('[loss]episode %d: %.2f' % (e, loss))

        if e % test_interval == 0 and not e == 0:
            scores = []
            for _ in range(100):
                terminate = False
                score = 0.
                state = env.reset()
                while not terminate:
                    state_one_hot = np.zeros(48)
                    state_one_hot[state[0] * 12 + state[1]] = 1.
                    state_one_hot = torch.Tensor(state_one_hot).to(device).unsqueeze(0)
                    action = model.get_action(state_one_hot)
                    next_state, reward, terminate = env.take_action(action)
                    score += reward
                    state = next_state
                scores.append(score)

            scores = np.asarray(scores)
            print(e, np.mean(scores))


def train_ActorCritic(env):
    model = ActorCritic(48, env.num_actions)

    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for e in range(1000):
        state = env.reset()

        terminate = False
        running_loss = 0.
        # create an episode
        while not terminate:
            state_one_hot = np.zeros(48)
            state_one_hot[state[0] * 12 + state[1]] = 1.
            state_one_hot = torch.Tensor(state_one_hot).to(device).unsqueeze(0)

            action = model.get_action(state_one_hot)
            next_state, reward, terminate = env.take_action(action)

            next_state_onehot = convert_state2onehot(next_state)

            mask = 0 if terminate else 1

            action_one_hot = torch.zeros(output_dim)
            action_one_hot[action] = 1
            transition = [state_one_hot, next_state_onehot, action, reward, mask]

            state = next_state
            loss = model.train_model(model, transition, optimizer)
            running_loss += loss

        print('[loss]episode %d: %.2f' % (e, running_loss))

        if e % test_interval == 0 and (not e == 0):
            scores = []
            model.eval()
            for i in range(100):
                terminate = False
                state = env.reset()
                state = torch.Tensor(state).to(device)
                state = state.unsqueeze(0)
                score = 0
                while not terminate:
                    action = model.get_action(state)
                    next_state, reward, terminate = env.take_action(action)
                    score += reward
                    state = next_state
                scores.append(score)
            model.train()
            print('[test score]episode %d: %.2f' % (e, np.mean(np.asarray(scores))))


if __name__ == '__main__':
    cw = CliffWalking()
    input_dim = 48
    output_dim = cw.num_actions
    print('state size:', input_dim)
    print('action size:', output_dim)

    # train_REINFORCE(cw)
    train_ActorCritic(cw)
    # test_cliff_warlking_by_hand(cw)
