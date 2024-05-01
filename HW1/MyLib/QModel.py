
import numpy as np

class Qmodel():

    def __init__(self, n_actions: int, n_states: tuple, alpha: float = 0.1, gamma: float =0.9) -> None:
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        table_dim = tuple([n for n in n_states] + [n_actions])
        self.q_table = np.zeros(table_dim)
        # self.next_q_table = np.zeros(table_dim)

    def predict(self, state: tuple) -> int:
        action = np.argmax(self.q_table[state], axis=-1)
        return action

    def update_table(self, state: tuple, action: int, reward: float, next_state: int):
        old_value = self.q_table[state + (action, )]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state + (action, )] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
