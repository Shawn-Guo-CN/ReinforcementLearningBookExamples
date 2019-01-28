import numpy as np


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
        trans_matrix = [[[]]*self.shape[1]] * self.shape[0]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for a in range(len(self.actions)):
                    trans_matrix[i][j].append(self._cal_new_position_((i, j), a))

        return trans_matrix

    def _cal_new_position_(self, old_pos, action):
        old_pos = np.asarray(old_pos)
        new_pos = old_pos + self.action2shift[action]
        new_pos[0] = min(new_pos[0], 0)
        new_pos[0] = max(new_pos[0], self.shape[0] - 1)
        new_pos[1] = min(new_pos[1], 0)
        new_pos[1] = max(new_pos[1], self.shape[1] - 1)
        if self.cliff[new_pos[0]][new_pos[1]]:
            return np.asarray([self.shape[0] - 1, 0])
        else:
            return new_pos

    def take_action(self):
        pass

    def show_pos(self):
        env = np.zeros(self.shape)
        env[self.pos[0]][self.pos[1]] = 1
        print(env)


if __name__ == '__main__':
    cw = CliffWalking()
    print(cw.transmit_tensor[3][0][:])
