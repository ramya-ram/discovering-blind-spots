import dawid_skene
import numpy as np
import pdb
import collections
import sys

"""
Majority vote baseline:
For each datapoint, collapse all labels into the majority label. Weight is 1 for all datapoints.
If a true 1 is received for a datapoint, the label is automatically set to 1 with weight 1.
E.g., x_1 = [0,0,0,1,1] would collapse to label = 0, weight = 1
      x_2 = [1,1,0,1,1] would collapse to label = 1, weight = 1
"""
def majority_vote(input_filename, output_filename):
    data, review_counts, classes = dawid_skene.read_input_data(input_filename)
    predicted_labels = collections.OrderedDict()
    for s in data:
        labels_list = data[s][1]
        if (1,'t') in labels_list:
            predicted_labels[s] = 1 # Label = 1 if a true 1 is received
        else:
            counts = np.zeros(len(classes))
            for label in labels_list:
                class_label, truth_label = label
                counts[class_label] += 1
            majority_label = np.argmax(counts)
            predicted_labels[s] = majority_label

    with open(output_filename, 'w') as f:
        for i in predicted_labels:
            f.write(""+str(list(i))+","+str(predicted_labels[i])+",1\n") # Weights are 1 for all datapoints

"""
Dummy baseline:
For each datapoint, collapse all labels into the majority label over all datapoints. Weight is 1 for all datapoints.
If a true 1 is received for a datapoint, the label is automatically set to 1 with weight 1.
E.g., If prior is majority 0's,
      x_1 = [0,0,0,1,1] would collapse to label = 0, weight = 1
      x_2 = [1,1,0,1,1] would collapse to label = 0, weight = 1
"""
def dummy(input_filename, output_filename):
    data, review_counts, classes = dawid_skene.read_input_data(input_filename)
    predicted_labels = collections.OrderedDict()
    total_counts = np.zeros(len(classes))
    for s in data:
        labels_list = data[s][1]
        for label in labels_list:
            class_label, truth_label = label
            total_counts[class_label] += 1
    for s in data:
        labels_list = data[s][1]
        if (1,'t') in labels_list:
            predicted_labels[s] = 1 # Label = 1 if a true 1 is received
        else:
            majority_label = np.argmax(total_counts)
            predicted_labels[s] = majority_label

    with open(output_filename, 'w') as f:
        for i in predicted_labels:
            f.write(""+str(list(i))+","+str(predicted_labels[i])+",1\n") # Weights are 1 for all datapoints

"""
All labels baseline:
For each datapoint, include every label as a separate datapoint. Weight is 1 for all datapoints.
If a true 1 is received for a datapoint, all other labels are removed and this datapoint's label is automatically set to 1 with weight 1.
E.g., x_1 = [0,0,0,1,1] would be converted to [(x_1,0), (x_1,0), (x_1,0), (x_1,1), (x_1,1)]
"""
def all_labels(input_filename, output_filename):
    data, review_counts, classes = dawid_skene.read_input_data(input_filename)
    with open(output_filename, 'w') as f:
        for s in data:
            labels_list = data[s][1]
            if (1,'t') in labels_list:
                f.write(""+str(list(s))+",1,1\n") # Label = 1 if a true 1 is received and weight = 1
                continue # Do not write multiple (1,'t')'s for a particular state - collapse to only one label
            else:
                for label in labels_list:
                    class_label, truth_label = label
                    f.write(""+str(list(s))+","+str(class_label)+",1\n")


if __name__ == '__main__':
    condition = sys.argv[1]
    globals().get(condition)(sys.argv[2], sys.argv[3])