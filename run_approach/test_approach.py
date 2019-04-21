import dawid_skene
import review
import classifier
import os
import sys
import matplotlib.pyplot as plt
import shutil
import random
import numpy as np
import math
import baselines
import gym
import yaml
from scipy import stats
import domains
sys.path.append('../run_q_learning')
from run_game import run
from q_learner import QLearner

"""
Usage:
python -W ignore test_approach.py <config-file>
This will run our approach to identify blind spots. More details about the format and content of the config file is included in the README.
e.g. python -W ignore test_approach.py run_config.yml
"""

# Label types is a list of tuples. The first element in the tuple is the name of the label type and the second element is a dictionary that specifies the characteristics of the label type.
all_label_types = [("random-acceptable",        {"data_bias":0, "SR_noise":1, "AM_noise":0}),
                   ("random-action-mismatch",   {"data_bias":0, "SR_noise":1, "AM_noise":1}),
                   ("demo-acceptable",          {"data_bias":1, "SR_noise":1, "AM_noise":0}),
                   ("demo-action-mismatch",     {"data_bias":1, "SR_noise":1, "AM_noise":1}),
                   ("corrections",              {"data_bias":2, "SR_noise":1, "AM_noise":0})]

class TestApproach:

    def __init__(self, sourceQ_file, targetQ_file, target_env):
        self.sourceQ_file = sourceQ_file
        self.targetQ_file = targetQ_file
        self.target_env = gym.make(target_env)
        self.label_types = []

    def test_all(self, save_dir, label_types, percentile, num_runs, budget_list, percent_sim_data, state_visits):
        self.label_types = label_types
        self.save_dir = save_dir
        self.percentile = percentile
        self.num_runs = num_runs
        self.filenames = {"sourceQ": self.sourceQ_file,
                          "targetQ": self.targetQ_file,
                          "sim_on_real": os.path.join(self.save_dir, "sim_on_real"),
                          "data": os.path.join(self.save_dir, "data.csv"),
                          "results": os.path.join(self.save_dir, "results.csv"),
                          "true_sim": os.path.join(self.save_dir, "true_sim_data.csv"),
                          "true_real": os.path.join(self.save_dir, "true_real_data.csv"),
                          "acceptable_actions": os.path.join(self.save_dir, "acceptable_actions.csv")}
        self.estimation_baselines = ["dawid_skene", "majority_vote"]
        self.classifier_baselines = ["dawid_skene", "majority_vote", "all_labels"]
        self.estimation_metrics = ["accuracy", "error", "error1s"]
        self.classifier_metrics = ["average_precision_score", "mean_squared_error","f1_score","accuracy_score","precision_score","recall_score"] #"roc_auc_score",
        self.test_data_list = ["seen","unseen","all"]
        self.oracle_in_loop_baselines = ["model_query", "always_query", "never_query"]
        self.oracle_in_loop_metrics = ["avg_reward","percent_queries"]
        self.estimation_results = {}
        self.classifier_results = {}
        self.oracle_in_loop_results = {}
        self.data_sizes = {}

        if len(state_visits) > 0:
            self.filenames["sim_on_real"] = state_visits
        else:
            agent = QLearner(self.target_env, sourceQ_file=self.sourceQ_file)
            run(self.target_env, agent, False, self.filenames["sim_on_real"], 10000)

        x_label = "Budget"
        x_list = budget_list

        for label_type in label_types:
            label = label_type[0]
            self.data_sizes[label] = -1

        # Creating data structures to store results
        for label_type in label_types:
            label = label_type[0]
            if label not in self.estimation_results:
                self.estimation_results[label] = {}
            for metric in self.estimation_metrics:
                if metric not in self.estimation_results[label]:
                    self.estimation_results[label][metric] = {}
                for baseline in self.estimation_baselines:
                    self.estimation_results[label][metric][baseline] = np.zeros((len(x_list), self.num_runs), dtype=np.float64)
        for label_type in label_types:
            label = label_type[0]
            if label not in self.classifier_results:
                self.classifier_results[label] = {}
            for metric in self.classifier_metrics:
                if metric not in self.classifier_results[label]:
                    self.classifier_results[label][metric] = {}
                for test_data in self.test_data_list:
                    if test_data not in self.classifier_results[label][metric]:
                        self.classifier_results[label][metric][test_data] = {}
                    for baseline in self.classifier_baselines:
                        self.classifier_results[label][metric][test_data][baseline] = np.zeros((len(x_list), self.num_runs), dtype=np.float64)
        for label_type in label_types:
            label = label_type[0]
            if label not in self.oracle_in_loop_results:
                self.oracle_in_loop_results[label] = {}
                for metric in self.oracle_in_loop_metrics:
                    if metric not in self.oracle_in_loop_results[label]:
                        self.oracle_in_loop_results[label][metric] = {}
                        for i in self.oracle_in_loop_baselines:
                            self.oracle_in_loop_results[label][metric][i] = np.zeros((len(x_list), self.num_runs), dtype=np.float64)

        # Run approach many times (based on num_runs) and with the whole range of budget values
        for num in range(self.num_runs):
            self.target_env.env.generate_training_subset(percent_sim_data)
            self.target_env.env.set_to_training_set()
            for i in range(len(x_list)):
                x = x_list[i]
                self.max_states = -1
                for label_type in label_types:
                    print(label_type," ",x)
                    self.test_one_instance(label_type, (i, x), (0, percent_sim_data), num)
            self.write_results(label_types, x_list, num)

    def write_results(self, label_types, x_list, run_num):
        estimation_filename = os.path.join(self.save_dir, 'results_estimation.csv')
        if os.path.exists(estimation_filename):
            os.remove(estimation_filename)
        with open(estimation_filename, 'a') as f:
            for label_type in label_types:
                label = label_type[0]
                f.write("label_type="+label+"\n")
                for metric in self.estimation_metrics:
                    f.write("\tmetric="+metric+"\n")
                    for i in self.estimation_baselines:
                        f.write("\t\tbaseline="+i+"\n")
                        all_y = self.estimation_results[label][metric][i]
                        mean_y = np.zeros(len(all_y))
                        sterr_y = np.zeros(len(all_y))
                        for g in range(len(all_y)):
                            mean_y[g] = np.mean(all_y[g,0:run_num+1])
                            sterr_y[g] = 0
                            if run_num > 0:
                                sterr_y[g] = stats.sem(all_y[g,0:run_num+1])
                        f.write("\t\t\tx="+(','.join(str(x) for x in x_list))+"\n")
                        f.write("\t\t\tmean="+(','.join(str(x) for x in mean_y))+"\n")
                        f.write("\t\t\tsterr="+(','.join(str(x) for x in sterr_y))+"\n")

        classifier_filename = os.path.join(self.save_dir, 'results_classifier.csv')
        if os.path.exists(classifier_filename):
            os.remove(classifier_filename)
        with open(classifier_filename, 'a') as f:
            f.write("Run num: "+str(run_num+1)+"\n") #+1 because 0-indexed
            for label_type in label_types:
                label = label_type[0]
                f.write("label_type="+label+"\n")
                metric = "f1_score"
                for t in range(len(self.test_data_list)):
                    test_data = self.test_data_list[t]
                    f.write("\tmetric="+test_data+"\n")
                    for i in self.classifier_baselines:
                        all_y = self.classifier_results[label][metric][test_data][i]
                        mean_y = np.zeros(len(all_y))
                        sterr_y = np.zeros(len(all_y))
                        for g in range(len(all_y)):
                            mean_y[g] = np.mean(all_y[g,0:run_num+1])
                            sterr_y[g] = 0
                            if run_num > 0:
                                sterr_y[g] = stats.sem(all_y[g,0:run_num+1])
                        f.write("\t\tbaseline="+i+"\n")
                        f.write("\t\t\tx="+(','.join(str(x) for x in x_list))+"\n")
                        f.write("\t\t\tmean="+(','.join(str(x) for x in mean_y))+"\n")
                        f.write("\t\t\tsterr="+(','.join(str(x) for x in sterr_y))+"\n")

        oracle_in_loop_filename = os.path.join(self.save_dir, 'results_OIL.csv')
        if os.path.exists(oracle_in_loop_filename):
            os.remove(oracle_in_loop_filename)
        with open(oracle_in_loop_filename, 'a') as f:
            for label_type in label_types:
                label = label_type[0]
                f.write("label_type="+label+"\n")
                for metric in self.oracle_in_loop_metrics:
                    f.write("\tmetric="+metric+"\n")
                    for i in self.oracle_in_loop_baselines:
                        f.write("\t\tbaseline="+i+"\n")
                        all_y = self.oracle_in_loop_results[label][metric][i]
                        mean_y = np.zeros(len(all_y))
                        sterr_y = np.zeros(len(all_y))
                        for g in range(len(all_y)):
                            mean_y[g] = np.mean(all_y[g,0:run_num+1])
                            sterr_y[g] = 0
                            if run_num > 0:
                                sterr_y[g] = stats.sem(all_y[g,0:run_num+1])
                        f.write("\t\t\tx="+(','.join(str(x) for x in x_list))+"\n")
                        f.write("\t\t\tmean="+(','.join(str(x) for x in mean_y))+"\n")
                        f.write("\t\t\tsterr="+(','.join(str(x) for x in sterr_y))+"\n")

    def test_one_instance(self, label_type, x_tuple, percent_tuple, run_num):
        x, x_value = x_tuple
        p, percent_sim_data = percent_tuple
        budget = x_value

        self.target_env.env.set_to_training_set()
        # Get data from oracle feedback
        num_seen_states, reviewed_blindspots = review.main(self.filenames, self.save_dir, self.target_env, label_type, budget, percent_sim_data, self.max_states, 0, self.percentile)
        if self.max_states == -1 or self.max_states < num_seen_states:
            self.max_states = num_seen_states

        prediction_probs_skene = None
        prediction_classes_skene = None
        for i in self.classifier_baselines:
            predicted_filename = os.path.join(self.save_dir, i+".csv")
            # Run approach given oracle data
            accuracy, mean_squared_error, error1s = self.run_approach(i, self.filenames["data"], predicted_filename, reviewed_blindspots, label_type)
            if i in self.estimation_baselines:
                self.estimation_results[label_type[0]]["accuracy"][i][x,run_num] = accuracy
                self.estimation_results[label_type[0]]["error"][i][x,run_num] = mean_squared_error
                self.estimation_results[label_type[0]]["error1s"][i][x,run_num] = error1s
            results, self.data_sizes[label_type[0]], prediction_probs, prediction_classes = classifier.main(self.save_dir, predicted_filename, self.filenames["true_sim"], label_type[0], i, self.classifier_metrics, self.filenames["sim_on_real"])
            if i == "dawid_skene":
                prediction_probs_skene = prediction_probs
                prediction_classes_skene = prediction_classes
            for metric in self.classifier_metrics:
                for t in self.test_data_list:
                    self.classifier_results[label_type[0]][metric][t][i][x,run_num] = results[t][metric]

        self.target_env.env.set_to_testing_set()
        for i in self.oracle_in_loop_baselines:
            probs = None
            classes = None
            if i == "model_query":
                probs = prediction_probs_skene
                classes = prediction_classes_skene
            # Run oracle-in-the-loop evaluation
            avg_reward, percent_queries = self.oracle_in_loop_eval(self.target_env, i, probs, classes)
            self.oracle_in_loop_results[label_type[0]]["avg_reward"][i][x,run_num] = avg_reward
            self.oracle_in_loop_results[label_type[0]]["percent_queries"][i][x,run_num] = percent_queries

    def run_approach(self, approach, data_filename, predicted_data_filename, reviewed_blindspots, label_type):
        # Run appropriate method (dawid-skene or baseline approaches)
        if approach == "dawid_skene":
            patient_classes, class_marginals, error_rates = dawid_skene.main(data_filename, predicted_data_filename, label_type[0])
            with open(self.filenames["results"], 'a') as f:
                f.write("\nApproach type: "+approach+"\n")
                f.write("Num distinct datapoints: "+str(len(patient_classes))+"\n")
                f.write("Class marginals\n"+str(class_marginals)+"\n")
                f.write("Error rates\n"+str(error_rates)+"\n")
        elif approach == "majority_vote":
            baselines.majority_vote(data_filename, predicted_data_filename)
        elif approach == "dummy":
            baselines.dummy(data_filename, predicted_data_filename)
        elif approach == "all_labels":
            baselines.all_labels(data_filename, predicted_data_filename)
            return -1, -1, -1

        # Accuracy on predicting true labels through aggregation
        accuracy, accuracy_1s, mean_squared_error, mean_squared_error_1s = self.get_skene_accuracy(predicted_data_filename, self.filenames["true_sim"], reviewed_blindspots)
        with open(self.filenames["results"], 'a') as f:
            if approach != "dawid_skene":
                f.write("\nApproach type: "+approach+"\n")
            f.write("Accuracy: "+str(accuracy)+"\n")
            f.write("Accuracy of 1s: "+str(accuracy_1s)+"\n")
            f.write("Mean squared error: "+str(mean_squared_error)+"\n")
            f.write("Mean squared error of 1s: "+str(mean_squared_error_1s)+"\n")
            f.write("Num of reviewed blindspots: "+str(len(reviewed_blindspots))+"\n")
        return accuracy, mean_squared_error, mean_squared_error_1s

    def get_skene_accuracy(self, predicted_file, true_file, reviewed_blindspots):
        predicted_data = review.read_labelled_file(predicted_file, True)
        true_data = review.read_labelled_file(true_file)

        num_total = 0
        num_correct = 0
        mean_squared_error = 0
        num_1s = 0
        num_1s_error = 0
        num_1s_correct = 0
        for s in predicted_data:
            predicted_label, predicted_weight = predicted_data[s]
            predicted_weight = float(predicted_weight)
            if predicted_label == 0:
                probOf1 = 1 - predicted_weight
            else:
                probOf1 = predicted_weight
            if true_data[s] == 1:
                num_1s += 1
                num_1s_error += math.pow(probOf1 - true_data[s], 2)
                if predicted_label == 1:
                    num_1s_correct += 1
            if predicted_label == true_data[s]:
                num_correct += 1
            mean_squared_error += math.pow(probOf1 - true_data[s], 2)
            num_total += 1

        accuracy = float(num_correct)/num_total
        mean_squared_error = float(mean_squared_error)/num_total
        mean_squared_error_1s = 0
        accuracy_1s = 0
        if num_1s > 0:
            mean_squared_error_1s = float(num_1s_error)/num_1s
            accuracy_1s = float(num_1s_correct)/num_1s
        return accuracy, accuracy_1s, mean_squared_error, mean_squared_error_1s

    def oracle_in_loop_eval(self, target_env, agent_type, prediction_probs, prediction_classes):
        # Gets performance on target task with agent querying the oracle for help
        # Compares an agent that uses our model to query, an agent that always queries, and an agent that never queries
        target_qtable, target_policy = review.read_Q(self.filenames["targetQ"])
        source_qtable, source_policy = review.read_Q(self.filenames["sourceQ"])
        num_episodes = 100
        curr_episode = 0
        num_total_queries = 0
        num_queries = 0
        total_reward = 0
        while curr_episode < num_episodes:
            state = target_env.reset()
            done = False
            episode_reward = 0
            while not done:
                state = tuple(state)
                # target_env.render()
                source_s = target_env.env.get_source_state(state)
                if (agent_type == "model_query" and prediction_classes[source_s] == 1) or agent_type == "always_query": #Blind spot state
                    action = target_policy[state]
                    num_queries += 1
                    # print("Query at state: ", state)
                    # input()
                elif (agent_type == "model_query" and prediction_classes[source_s] == 0) or agent_type == "never_query":
                    action = source_policy[source_s]
                next_state, reward, done, _info = target_env.step(action)
                num_total_queries += 1
                total_reward += reward
                episode_reward += reward
                state = next_state
            # print(episode_reward)
            curr_episode += 1
        avg_reward = float(total_reward)/num_episodes
        percent_queries = float(num_queries)/num_total_queries
        return avg_reward, percent_queries

def get_label_type(label_str):
    for label_type in all_label_types:
        if label_str == label_type[0]:
            return label_type

if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    save_dir = cfg["save_dir"]
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    shutil.copy(config_file, os.path.join(save_dir, "config.yml"))

    label_types = []
    for i in cfg["label_types"]:
        label_types.append(get_label_type(i))
    test_approach = TestApproach(cfg["sourceQ_file"], cfg["targetQ_file"], cfg["target_env"])
    test_approach.test_all(save_dir, label_types, cfg["percentile"], cfg["num_runs"], eval(cfg["budget_list"]), cfg["percent_data"], cfg["state_visits"])