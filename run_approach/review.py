import os
import numpy as np
import sys
import collections
import random
import gym
import seaborn as sns
import matplotlib.pyplot as plt

class Review:
    def __init__(self, filenames, file_dir, env, label_type, budget, percent_sim_data, max_states, percentage_review, percentile):
        self.filenames = filenames
        self.file_dir = file_dir
        self.env = env
        self.feature_bins = self.env.env.feature_bins
        self.label_type = label_type
        self.budget = int(budget)
        self.percent_sim_data = float(percent_sim_data)
        self.max_states = int(max_states)
        self.percentage_review = float(percentage_review)
        self.percentile = float(percentile)
        self.acceptable_id = 0
        self.unacceptable_id = 1

    # The last two arguments (acceptable_actions and optimal_policy) are needed for corrections
    def get_trajectory_data(self, label_type, env, feature_bins, policy, budget, acceptable_actions=None, optimal_policy=None):
        data = {}
        num_collected = 0
        while num_collected < budget:
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                state = tuple(state)
                # env.render()
                if label_type[1]["data_bias"] == 2: # Corrections
                    source_s = self.env.env.get_source_state(state)
                    action = policy[source_s]
                    if source_s not in data:
                        data[source_s] = []
                    if action in acceptable_actions[state]:
                        # If the agent's action is acceptable, the oracle does not correct the action.
                        data[source_s].append((self.acceptable_id, 't'))
                    else:
                        # If the agent's action is unacceptable, the oracle corrects the action. The agent continues with the corrected action but notes the unacceptable action at that state.
                        data[source_s].append((self.unacceptable_id, 't'))
                        if label_type[1]["data_bias"] == 2:
                            action = optimal_policy[state]

                elif label_type[1]["data_bias"] == 1: # Demonstrations
                    action = policy[state]
                    if state not in data:
                        data[state] = []
                    data[state].append(action)

                num_collected += 1
                next_state, reward, done, _info = env.step(action)
                state = next_state
                total_reward += reward
            # print(total_reward)
        return data

    def check_action_mismatches(self, agent_policy, expert_data):
        all_review_states = {}
        data = {}
        source_states = []
        for s in expert_data:
            expert_actions = expert_data[s]
            source_s = self.env.env.get_source_state(s) # Map target world state into appropriate source state (in this case, the last feature is removed)
            if source_s not in source_states:
                source_states.append(source_s)
            agent_action = agent_policy[source_s]
            for expert_action in expert_actions:
                if agent_action != expert_action:
                    if source_s not in all_review_states:
                        all_review_states[source_s] = []
                    all_review_states[source_s].append(s) # If agent action is not in human action set, this state becomes a possible review state
                else:
                    if source_s not in data:
                        data[source_s] = []
                    data[source_s].append((self.acceptable_id, 't')) # If agent action is in the human action set for this state, this state is automatically acceptable
        return data, all_review_states

    def full_review(self, source_policy, expert_data):
        data, all_review_states = self.check_action_mismatches(source_policy, expert_data)
        acceptable_actions = read_list_labelled_file(self.filenames["acceptable_actions"])
        for source_s in all_review_states:
            if source_s not in data:
                data[source_s] = []
            for target_s in all_review_states[source_s]:
                # All data points are "true" because there is full review (there will still be noisy 0's because of state representation mismatch)
                data[source_s].append(self.get_reviewed_label(source_policy, source_s, acceptable_actions, target_s))
        return data

    def no_review(self, source_policy, expert_data):
        data, all_review_states = self.check_action_mismatches(source_policy, expert_data)
        for source_s in all_review_states:
            if source_s not in data:
                data[source_s] = []
            for target_s in all_review_states[source_s]:
                data[source_s].append((self.unacceptable_id, 'n')) # No budget for review so all states with action mismatches are considered a "noisy" unacceptable
        total_labels = 0
        for s in data:
            total_labels += len(data[s])
        return data

    def get_reviewed_label(self, source_policy, source_s, acceptable_actions, target_s):
        if source_policy[source_s] in acceptable_actions[target_s]:
            return (0, 't')
        else:
            return (1, 't')

    def run_review(self):
        print(self.label_type)
        if self.label_type[0] == 'perfect_oracle': # Get perfect labels of simulator states
            all_data = read_labelled_file(self.filenames["true_sim"])
            if self.max_states > 0:
                num_subset = self.max_states
            else:
                num_subset = int(self.percent_sim_data*len(all_data))
            subset_states = random.sample(all_data.keys(), num_subset)
            data = {}
            for s in subset_states:
                data[s] = [(all_data[s],'t')]
        else:
            target_qtable, target_policy = read_Q(self.filenames["targetQ"])
            source_qtable, source_policy = read_Q(self.filenames["sourceQ"])
            acceptable_actions = read_list_labelled_file(self.filenames["acceptable_actions"])

            # Trajectory data (corrections)
            if self.label_type[1]["data_bias"] == 2 or self.label_type[1]["data_bias"] == 5:
                data = self.get_trajectory_data(self.label_type, self.env, self.feature_bins, source_policy, self.budget, acceptable_actions=acceptable_actions, optimal_policy=target_policy)
            else:
                all_data = read_labelled_file(self.filenames["true_real"])
                sim_data = read_labelled_file(self.filenames["true_sim"])
                # Random data
                if self.label_type[1]["data_bias"] == 0:
                    to_review_data = self.get_random_data(sim_data, self.percent_sim_data, self.max_states, all_data, self.budget, target_qtable, target_policy)

                # Trajectory data (demonstrations)
                elif self.label_type[1]["data_bias"] == 1:
                    to_review_data = self.get_trajectory_data(self.label_type, self.env, self.feature_bins, target_policy, self.budget)

                # Amount of review
                if self.label_type[1]["AM_noise"] == 0:
                    data = self.full_review(source_policy, to_review_data)

                elif self.label_type[1]["AM_noise"] == 1:
                    data = self.no_review(source_policy, to_review_data)

        reviewed_blindspots = []
        with open(self.filenames["data"], 'w') as f:
            for s in data:
                f.write(str(list(s))+"=[")
                for i in range(len(data[s])):
                    label = data[s][i]
                    class_label = label[0]
                    truth_label = label[1]
                    if class_label == 1 and truth_label == 't' and s not in reviewed_blindspots:
                        reviewed_blindspots.append(s)
                    f.write("("+str(class_label)+";"+str(truth_label)+")")
                    if i < len(data[s])-1:
                        f.write(",")
                f.write("]\n")
        true_data =  read_labelled_file(self.filenames["true_sim"])
        with open(os.path.join(self.file_dir, "true_data.csv"), 'w') as f:
            for s in data:
                f.write(str(list(s))+","+str(true_data[s])+"\n")

        # Confusion counts provide information on how noisy the labels are, given the true labels
        self.calculate_confusion_counts(self.filenames["data"], self.filenames["true_sim"])
        return len(data), reviewed_blindspots

    def get_random_data(self, sim_data, percent_sim_data, max_states, all_data, num_sample, target_qtable, target_policy):
        if max_states > 0:
            num_subset = max_states
        else:
            num_subset = int(percent_sim_data*len(sim_data))
        to_review_states = []

        states_list, weights_list = self.env.env.get_uniform_state_weights()
        subset_states_sim = []
        for i in range(num_sample):
            index = np.random.choice(range(len(states_list)), 1, p=weights_list)[0]
            state = states_list[index]
            source_s = self.env.env.get_source_state(state)
            if len(subset_states_sim) < num_subset:
                if source_s not in subset_states_sim:
                    subset_states_sim.append(source_s)
            else:
                while source_s not in subset_states_sim:
                    index = np.random.choice(range(len(states_list)), 1, p=weights_list)[0]
                    state = states_list[index]
                    source_s = self.env.env.get_source_state(state)
            to_review_states.append(state)

        to_review_data = {}
        for s in to_review_states:
            if s not in to_review_data:
                to_review_data[s] = []
            to_review_data[s].append(target_policy[s])
        return to_review_data


    # Observe what the noise of the data looks like
    def calculate_confusion_counts(self, label_filename, true_filename):
        noisy_data = read_list_labelled_file(label_filename)
        true_data =  read_labelled_file(true_filename)
        prior = [0,0]
        confusion_matrix=[[0,0],[0,0]]
        counts_per_state = {0:[],1:[]}
        for i in noisy_data:
            noisy_labels = noisy_data[i]
            if type(noisy_labels) == int:
                noisy_labels = [noisy_labels]
            true_label = true_data[i]
            prior[true_label] += 1
            counts = [0,0]
            for noisy_label in noisy_labels:
                confusion_matrix[true_label][noisy_label] += 1
                counts[noisy_label] += 1
            counts_per_state[true_label].append((i,counts))
        with open(self.filenames["results"], 'a') as f:
            f.write("\n-------------------\n")
            f.write("Label type: "+self.label_type[0]+", Budget: "+str(self.budget)+", Percentage review: "+str(self.percentage_review)+ \
                ", Percent of sim data: "+str(self.percent_sim_data)+", Max states: "+str(self.max_states)+"\n")
            f.write("TargetQ_file: "+self.filenames["targetQ"]+"\n")
            f.write("Percentile: "+str(self.percentile)+"\n\n")
            f.write("Prior and confusion matrix from this game configuration:"+"\n")
            f.write("Prior counts:\n"+str(prior)+"\n")
            f.write("Confusion matrix counts:\n"+str(confusion_matrix)+"\n")
        if sum(prior) > 0:
            prior = [float(i)/sum(prior) for i in prior]
            prior = [round(i, 2) for i in prior]
        for index in range(len(confusion_matrix)):
            if sum(confusion_matrix[index]) > 0:
                confusion_matrix[index] = [float(i)/sum(confusion_matrix[index]) for i in confusion_matrix[index]]
                confusion_matrix[index] = [round(i, 2) for i in confusion_matrix[index]]
        with open(self.filenames["results"], 'a') as f:
            f.write("Normalized prior:\n"+str(prior)+"\n")
            f.write("Normalized confusion matrix:\n"+str(confusion_matrix)+"\n")
            # Print 0 and 1 observation counts for each safe and blindspot state
        #     for i in counts_per_state:
        #         f.write(str(i)+":\n")
        #         for j in counts_per_state[i]:
        #             f.write(str(j)+"\n")

    def generate_true_blind_spots(self, percentile):
        target_qtable, target_policy = read_Q(self.filenames["targetQ"])
        source_qtable, source_policy = read_Q(self.filenames["sourceQ"])
        deltas = []
        for s in target_qtable:
            action_values = target_qtable[s][:]
            action_values.sort()
            # All mistakes
            for i in range(len(action_values)-1):
                delta = action_values[len(action_values)-1] - action_values[i]
                if delta > 0:
                    deltas.append(delta)

        deltas.sort()
        percentile = float(percentile)
        if percentile > 0:
            if int(len(deltas)*percentile) < 0 or int(len(deltas)*percentile) >= len(deltas):
                pdb.set_trace()
            cutoff_delta = deltas[int(len(deltas)*percentile)]
        else:
            cutoff_delta = 0

        plot_name = os.path.join(self.file_dir, "percentile_graph.png")
        if not os.path.exists(plot_name):
            plt.figure()
            n, bins, patches = plt.hist(deltas,25)
            if percentile > 0:
                plt.title("Cutoff delta: "+str(format(cutoff_delta,'.2f')))
                patch_num = np.digitize(cutoff_delta, bins)
                for i in range(patch_num, len(patches)):
                    patches[i].set_fc('r')
            else:
                plt.title("Only optimal actions are acceptable")
            plt.savefig(plot_name)

        target_blind_spots = []
        source_blind_spots = []
        acceptable_actions = collections.OrderedDict()
        num_blind_spots_right = 0
        num_blind_spots_left = 0
        all_source_states = []
        if percentile == -1:
            for s in target_policy:
                action_values = target_qtable[s]
                num_actions = len(action_values)
                if s not in acceptable_actions:
                    acceptable_actions[s] = []
                acceptable_actions[s].append(target_policy[s])
                source_s = self.env.env.get_source_state(s)
                if source_s not in all_source_states:
                    all_source_states.append(source_s)
                if source_policy[source_s] not in acceptable_actions[s]:
                    target_blind_spots.append(s)
                    source_blind_spots.append(source_s)
        else:
            num_actions = 0
            for s in target_qtable:
                action_values = target_qtable[s]
                source_s = self.env.env.get_source_state(s)
                num_actions = len(action_values)
                if source_s in source_policy:
                    if source_s not in all_source_states:
                        all_source_states.append(source_s)
                    if s not in acceptable_actions:
                        acceptable_actions[s] = []
                    for i in range(len(action_values)):
                        if (action_values[target_policy[s]] - action_values[i]) <= cutoff_delta:
                            acceptable_actions[s].append(i) # acceptable action
                    if (action_values[target_policy[s]] - action_values[source_policy[source_s]]) > cutoff_delta:
                        target_blind_spots.append(s)
                        source_blind_spots.append(source_s)

        write_ground_truth(self.filenames["true_real"], target_qtable, target_blind_spots)
        write_ground_truth(self.filenames["true_sim"], source_qtable, source_blind_spots)
        with open(self.filenames["acceptable_actions"], 'w') as f:
            for s in acceptable_actions:
                f.write(str(list(s))+"="+str(list(acceptable_actions[s]))+"\n")

def read_list_labelled_file(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    data = {}
    for line in lines:
        values = line.split("=")
        x_i = values[0][1:-1].split(",")
        x_i = tuple([int(x.strip()) for x in x_i])
        if x_i not in data:
            data[x_i] = []
        labels = values[1][1:-1].split(",") # List of labels
        for label_str in labels:
            if "(" in label_str:
                label_values = label_str[1:-1].split(";")
                class_label = int(label_values[0])
                truth_label = label_values[1]
                label = (class_label, truth_label)
                data[x_i].append(class_label)
            else:
                data[x_i].append(int(label_str))
    return data

def read_labelled_file(filename, save_extra_info=False):
    data = {}
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        line = line[1:] # Remove first bracket
        values = line.split("],")
        if "," in values[0]:
            s = values[0].split(",")
            s = tuple([int(x) for x in s])
        else:
            s = int(values[0])

        weight = None
        if "," in values[1]:
            info = values[1].split(",") # Data may have additional pieces of information in addition to the label. Ignore for now.
            label = int(info[0])
            if save_extra_info and len(info[1]) > 0:
                weight = info[1]
        else:
            label = int(values[1])

        if s not in data:
            data[s] = []

        if weight != None:
            data[s].append((label,weight))
        else:
            data[s].append(label)
    for s in data:
        if len(data[s]) == 1:
            data[s] = data[s][0] # Convert from list to int if only one element
    return data

def read_Q(filename):
    qtable = collections.OrderedDict()
    policy = collections.OrderedDict()
    # Read in learned Q-value function
    lines = [line.rstrip('\n') for line in open(filename)]
    last_s = (-1)
    for line in lines:
        line = line[1:] # Remove first bracket
        values = line.split(']')
        s = values[0].split(',')
        s = [int(x) for x in s]

        s = tuple(s)
        action_value = values[1][1:].split(',')
        a = int(action_value[0])
        if last_s != s:
            qtable[s] = []
        qtable[s].append(float(action_value[1]))
        last_s = s

    # Obtain policy from Q-value function
    for s in qtable:
        max_value = qtable[s][0]
        max_action = 0
        for j in range(len(qtable[s])):
            if qtable[s][j] >= max_value:
                max_value = qtable[s][j]
                max_action = j
        policy[s] = max_action
    return qtable, policy

def write_ground_truth(filename, qtable, blind_spot_states):
    with open(filename, 'w') as f:
        for s in qtable:
            if s in blind_spot_states:
                blindspot_label = 1
            else:
                blindspot_label = 0
            f.write(str(list(s))+","+str(blindspot_label)+"\n")

def main(filenames, file_dir, env, label_type, budget, percent_sim_data, max_states, percentage_review, percentile):
    review = Review(filenames, file_dir, env, label_type, budget, percent_sim_data, max_states, percentage_review, percentile)
    review.generate_true_blind_spots(percentile)
    return review.run_review()