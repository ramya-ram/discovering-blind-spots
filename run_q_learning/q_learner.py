import numpy as np
import pdb
import os
import domains

"""
Run tabular Q-learning on the provided environment.
"""
class QLearner:

    def __init__(self, env, sourceQ_file=None,
                 gamma=0.9, init_lr=0.5, final_lr=0.1,
                 init_e=1.0, final_e=0.1, exploration_anneal_episodes=1000000):
        self.env = env.env
        self.feature_bins = env.env.feature_bins
        self.action_dim = env.action_space.n
        self.actions = list(range(env.action_space.n))
        self.gamma = gamma
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.init_e = init_e
        self.final_e = final_e
        self.e = self.init_e
        self.lr = self.init_lr
        self.exploration_anneal_episodes = exploration_anneal_episodes
        self.sourceQ_file = sourceQ_file
        if self.sourceQ_file != None:
            init_e = 0
        self.build_model()

    def build_model(self):
        # Build q-table
        flat_state_size = 1
        for i in range(len(self.feature_bins)):
            flat_state_size = flat_state_size * len(self.feature_bins[i])
        self.qtable = np.zeros((flat_state_size, self.action_dim))
        self.state_counts = np.zeros(flat_state_size)
        self.state_action_counts = np.zeros((flat_state_size, self.action_dim))
        self.state_Q_deltas = {}
        self.states = []
        self.learnedQ_state_dim = -1

        if self.sourceQ_file != None:
            lines = [line.rstrip('\n') for line in open(self.sourceQ_file)]
            for line in lines:
                line = line[1:] # Remove first bracket
                values = line.split(']')
                s = values[0].split(',')
                s = [int(x) for x in s]
                if s not in self.states:
                    self.states.append(s)
                self.learnedQ_state_dim = len(s)
                flat_s = self.flatten_state(s)
                action_value = values[1][1:].split(',')
                a = int(action_value[0])
                self.qtable.itemset((flat_s,a), action_value[1])
            self.e = 0
            self.lr = 0
        else:
            self.states = self.env.get_states()
        return self.qtable

    def flatten_state(self, state):
        # If target state is different from source state, map target to source state
        if self.sourceQ_file != None and len(state) != self.learnedQ_state_dim:
            state = self.env.get_source_state(state)
        # Flatten state into unique id
        state = list(state)
        flat_state = 0
        total_values = 1
        for i in range(len(state)):
            flat_state = flat_state + total_values * (np.digitize(state[i], self.feature_bins[i])-1)
            total_values = total_values * len(self.feature_bins[i])
        return flat_state

    def saveQ(self, save_dir):
        filename = os.path.join(save_dir, "Q.csv")
        if len(self.qtable) > 0:
            with open(filename, 'w') as f:
                for state in self.states:
                    flat_state = self.flatten_state(state)
                    print_state = state
                    if isinstance(state[0], float):
                        print_state = [float(format(i,'.2f')) for i in state]
                    for a in range(len(self.qtable[flat_state])):
                        f.write(str(list(print_state))+","+str(a)+","+str(self.qtable[flat_state][a])+"\n")

    def save_debug_info(self, save_dir):
        with open(os.path.join(save_dir, "state_counts.csv"),'w') as f:
            for state in self.states:
                flat_state = self.flatten_state(state)
                f.write(str(list(state))+","+str(int(self.state_counts[flat_state]))+"\n")

        with open(os.path.join(save_dir, "state_action_counts.csv"),'w') as f:
            for state in self.states:
                flat_state = self.flatten_state(state)
                for a in range(len(self.state_action_counts[flat_state])):
                    f.write(str(list(state))+","+str(a)+","+str(self.state_action_counts[flat_state][a])+"\n")

        if len(self.state_Q_deltas) >= 100:
            with open(os.path.join(save_dir, "state_deltas.csv"),'a') as f:
                total_deltas = 0
                for state in self.states:
                    flat_state = self.flatten_state(state)
                    if flat_state not in self.state_Q_deltas:
                        continue
                    prev_delta = 0
                    deltas = self.state_Q_deltas[flat_state]
                    if len(deltas) > 1:
                        prev_delta = deltas[len(deltas)-2]
                    total_deltas += abs(deltas[len(deltas)-1] - prev_delta)
                f.write(str(total_deltas)+"\n")

    def select_action(self, state):
        # Epsilon-greedy action selection
        if self.e > np.random.rand():
            action = np.random.choice(self.actions)
        else:
            flat_state = self.flatten_state(state)
            values = self.qtable[flat_state, :]
            action = np.random.choice(np.flatnonzero(values == values.max())) # Randomly pick from actions with maximum value
        return action

    def update_params(self):
        # Update epsilon and learning rate
        if self.e > 0:
            self.e = max(self.e -
                         (self.init_e - self.final_e) /
                         float(self.exploration_anneal_episodes),
                         self.final_e)
        if self.lr > 0:
            self.lr = max(self.lr -
                         (self.init_lr - self.final_lr) /
                         float(self.exploration_anneal_episodes),
                         self.final_lr)
        return self.e, self.lr

    def updateQ(self, state, action, reward, next_state):
        # Update Q-values and parameters (epsilon, learning rate, etc.)
        flat_state = self.flatten_state(state)
        next_flat_state = self.flatten_state(next_state)
        Q_state_action = self.qtable[flat_state, action]
        Q_next_state = self.qtable[next_flat_state, :]
        Q_next_state_max = np.amax(Q_next_state)
        loss = (reward + self.gamma * Q_next_state_max - Q_state_action)
        self.qtable[flat_state, action] = Q_state_action +  self.lr * loss
        self.update_params()

        # Store debug information
        num_states_deltas = 100
        max_value = max(self.qtable[flat_state, :])
        min_value = min(self.qtable[flat_state, :])
        self.state_counts[flat_state] += 1
        self.state_action_counts[flat_state][action] += 1
        if len(self.state_Q_deltas) < num_states_deltas:
            if flat_state not in self.state_Q_deltas:
                self.state_Q_deltas[flat_state] = []
        if flat_state in self.state_Q_deltas:
            self.state_Q_deltas[flat_state].append((max_value-min_value))