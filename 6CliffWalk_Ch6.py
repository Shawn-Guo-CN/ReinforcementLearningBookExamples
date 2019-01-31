import numpy as np
from collections import namedtuple, deque


gamma = 1.0
alpha = 0.5
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))
log_internal = 2
np.random.seed(2333)


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

        # game rules added by Shawn
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
            terminate = False

        if old_pos[0] == self.shape[0] - 1 and old_pos[1] == self.shape[1] - 1:
            terminate = True
            reward = 0.
            new_pos[0] = self.shape[0] - 1
            new_pos[1] = self.shape[1] - 1

        return new_pos, reward, terminate

    def take_action(self, action):
        self.steps += 1
        if self.steps > self.step_limit:
            return self.pos, -100., True
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
        self.steps = 0
        self.pos = np.asarray([self.shape[0] - 1, 0])
        return self.pos


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


class Q_net(object):
    def __init__(self, shape=[4, 12], num_actions=4, epsilon=1e-2):
        self.epsilon = epsilon

        self.q_est = [[[] for _ in range(shape[1])] for __ in range(shape[0])]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for a in range(num_actions):
                    self.q_est[i][j].append(0.)

    def get_action(self, state, test=False):
        state = state.tolist()
        if np.random.uniform() > self.epsilon or test:
            return np.argmax(self.q_est[state[0]][state[1]])
        else:
            return np.random.randint(4)

    @classmethod
    def train_by_sarsa(cls, model, transition, gamma=1.0, alpha=0.5):
        state, next_state, action, reward = transition
        next_action = model.get_action(state)

        model.q_est[state[0]][state[1]][action] += alpha * (reward
            + gamma * model.q_est[next_state[0]][next_state[1]][next_action]
                    - model.q_est[state[0]][state[1]][action])

    @classmethod
    def train_by_q_learning(cls, model, transition, gamma=1.0, alpha=0.5):
        state, next_state, action, reward = transition
        next_action = model.get_action(state, test=True)

        model.q_est[state[0]][state[1]][action] += alpha * (reward
                                                            + gamma * model.q_est[next_state[0]][next_state[1]][
                                                                next_action]
                                                            - model.q_est[state[0]][state[1]][action])


def train_sarsa(env, model):
    for e in range(3000):
        state = env.reset()

        terminate = False
        # create an episode
        while not terminate:
            action = model.get_action(state)
            next_state, reward, terminate = env.take_action(action)
            transition = [state, next_state, action, reward]
            state = next_state

            model.train_by_sarsa(model, transition)

        if e % log_internal == 0 and not e == 0:
            rewards = []
            for _ in range(100):
                terminate = False
                total_reward = 0.
                state = env.reset()
                while not terminate:
                    action = model.get_action(state, test=True)
                    next_state, reward, terminate = env.take_action(action)
                    total_reward += reward
                    state = next_state
                rewards.append(total_reward)

            rewards = np.asarray(rewards)
            print(e, np.mean(rewards))


def train_q_learning(env, model):
    for e in range(3000):
        state = env.reset()

        terminate = False
        # create an episode
        while not terminate:
            action = model.get_action(state)
            next_state, reward, terminate = env.take_action(action)
            transition = [state, next_state, action, reward]
            state = next_state

            model.train_by_q_learning(model, transition)

        if e % log_internal == 0 and not e == 0:
            rewards = []
            for _ in range(100):
                terminate = False
                total_reward = 0.
                state = env.reset()
                while not terminate:
                    action = model.get_action(state, test=True)
                    next_state, reward, terminate = env.take_action(action)
                    total_reward += reward
                    state = next_state
                rewards.append(total_reward)

            rewards = np.asarray(rewards)
            print(e, np.mean(rewards))


if __name__ == '__main__':
    cw = CliffWalking()
    q_net = Q_net()
    # train_sarsa(cw, q_net)
    train_q_learning(cw, q_net)
    # test_cliff_warlking_by_hand(cw)
