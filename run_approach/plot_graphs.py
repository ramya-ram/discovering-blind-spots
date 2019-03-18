import sys
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# Obtain graphs comparing our model to baseline approaches. The metric used is F1-scores.
def plot_classifier(results, colors, save_dir, prefix):
    for label_type in results:
        for t in results[label_type]:
            # If you want to plot just on states seen by the agent in oracle data collection:
            # Otherwise, comment out the two lines below
            if t != "seen":
                continue
            plt.figure()
            for i in results[label_type][t]:
                label_name = i
                x = np.array(results[label_type][t][i]["x"])
                mean_y = np.array(results[label_type][t][i]["mean"])
                sterr_y = np.array(results[label_type][t][i]["sterr"])
                plt.plot(x, mean_y, label = label_name, linestyle = '-', linewidth = 2, color=colors[i])
                plt.fill_between(x, mean_y-sterr_y, mean_y+sterr_y, alpha=0.2, edgecolor=colors[i], facecolor=colors[i])
            plt.legend(loc=0, fontsize=16)
            plt.ylabel("F1-score", fontsize=16)
            plt.xlabel("Budget", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # If you want to plot seen, unseen, and all:
            # plt.savefig(os.path.join(save_dir, prefix+"_"+label_type+"_"+t+".png"))
            # If you want to plot just on states seen by the agent in oracle data collection:
            plt.savefig(os.path.join(save_dir, prefix+"_"+label_type+".png"))

# Obtain graphs for oracle-in-the-loop evaluation, which compares an agent using our model to query, an agent that always queries, and an agent that never queries.
# The metric used is reward on the target task.
def plot_OIL(results, colors, save_dir):
    for label_type in results:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        for i in results[label_type]["avg_reward"]:
            x = np.array(results[label_type]["avg_reward"][i]["x"])
            mean_y = np.array(results[label_type]["avg_reward"][i]["mean"])
            sterr_y = np.array(results[label_type]["avg_reward"][i]["sterr"])
            ax1.plot(x, mean_y, label = i, linestyle = '-', linewidth = 2, color=colors[i])
            ax1.fill_between(x, mean_y-sterr_y, mean_y+sterr_y, alpha=0.2, edgecolor=colors[i], facecolor=colors[i])

        x = np.array(results[label_type]["percent_queries"]["model_query"]["x"])
        mean_y = np.array(results[label_type]["percent_queries"]["model_query"]["mean"])
        sterr_y = np.array(results[label_type]["percent_queries"]["model_query"]["sterr"])
        ax2.plot(x, mean_y, label = "percent_queries", linestyle = '-.', linewidth = 3, color='green')

        loc = 0
        ax1.legend(loc=loc, fontsize=16)
        ax2.legend(loc=0, fontsize=16)
        ax1.set_xlabel("Budget",fontsize=16)
        ax1.set_ylabel("Reward",fontsize=16)
        ax1.set_yticks(np.arange(-100,100,50))
        ax2.set_ylabel("Percentage of times queried",fontsize=16, color='green')
        ax2.set_yticks(np.arange(0,1.2,0.2))
        plt.savefig(os.path.join(save_dir, "oil_"+label_type+".png"))

def read_results(filename):
    results = {}
    curr_label = None
    curr_metric = None
    curr_baseline = None
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        if "Run num:" in line:
            continue
        elif "label_type" in line:
            curr_label = line.strip("\t").split("=")[1]
            if curr_label not in results:
                results[curr_label] = {}
        elif "metric" in line:
            curr_metric = line.strip("\t").split("=")[1]
            if curr_metric not in results[curr_label]:
                results[curr_label][curr_metric] = {}
        elif "baseline" in line:
            curr_baseline = line.strip("\t").split("=")[1]
            if curr_baseline not in results[curr_label][curr_metric]:
                results[curr_label][curr_metric][curr_baseline] = {}
        else:
            values = line.strip("\t").split("=")
            results[curr_label][curr_metric][curr_baseline][values[0]] = [float(i) for i in values[1].split(",")]
    return results

if __name__ == '__main__':
    file_dir = sys.argv[1]
    colors = {"dawid_skene":'b',"all_labels":'r',"majority_vote":'g',
              "model_query":'b',"always_query":'purple',"never_query":'r'}
    results_OIL = read_results(os.path.join(file_dir,"results_OIL.csv"))
    plot_OIL(results_OIL, colors, file_dir)
    # If you only want final plots with F1-scores:
    results_classifier = read_results(os.path.join(file_dir,"results_classifier.csv"))
    plot_classifier(results_classifier, colors, file_dir, "cf")

    # If you want plots with aggregation and classifier performance
    # results_classifier = read_results(os.path.join(file_dir,"results_classifier.csv"))
    # plot_classifier(results_classifier, colors, file_dir, "rf")
    # results_estimation = read_results(os.path.join(file_dir,"results_estimation.csv"))
    # plot_classifier(results_estimation, colors, file_dir, "ds")