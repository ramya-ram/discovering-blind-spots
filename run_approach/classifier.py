import numpy as np
from sklearn import ensemble, metrics
import collections
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterSampler
from sklearn.utils import shuffle
import review
from sklearn.base import clone
import os

class Classifier:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.data = []
        self.results = collections.OrderedDict()
        self.classifier = ensemble.RandomForestClassifier()
        self.params = {"criterion": ["gini", "entropy"],
                       "min_samples_split": [2, 10, 20],
                       "max_depth": [None, 2, 5, 10],
                       "min_samples_leaf": [1, 5, 10],
                       "max_leaf_nodes": [None, 5, 10, 20],
                       "n_estimators": [10, 100]}

    def read_data(self, filename):
        lines = [line.rstrip('\n') for line in open(filename)]
        x = []
        y = []
        weights = []
        for line in lines:
            values = line[1:].split("],") # Remove initial "[", then split at "]," to get the state separated from the label
            x_value = values[0].split(",")
            x_value = [int(x.strip()) for x in x_value]
            x.append(x_value)
            v = values[1].split(",")
            y.append(int(v[0]))
            if len(v) > 1:
                weights.append(float(v[1]))
        return x,y,weights

    def train_test(self, training_filename, testing_filename, state_visits, label, approach_type, all_metric_scores):
        self.state_visits = state_visits
        train_x, train_y, train_weights = self.read_data(training_filename)
        test_x, test_y, test_weights = self.read_data(testing_filename)
        test_data = {'seen': (),'unseen': (),'all': ()}
        test_weights = []
        unseen_x = []
        unseen_y = []
        unseen_weights = []
        seen_x = []
        seen_y = []
        seen_weights = []
        # Create multiple datasets: one for seen states, one for unseen states, and one for all states
        for i in range(len(test_x)):
            test_weight = state_visits[tuple(test_x[i])]
            if test_x[i] in train_x:
                seen_x.append(test_x[i])
                seen_y.append(test_y[i])
                seen_weights.append(test_weight)
            else:
                unseen_x.append(test_x[i])
                unseen_y.append(test_y[i])
                unseen_weights.append(test_weight)
            test_weights.append(test_weight)
        test_data['all'] = (test_x, test_y, test_weights)
        test_data['seen'] = (seen_x, seen_y, seen_weights)
        test_data['unseen'] = (unseen_x, unseen_y, unseen_weights)

        self.data_sizes = {}
        # As long as not all training examples are of the same class, train a model
        if train_y.count(train_y[0]) != len(train_y):
            model, best_threshold = self.train_full_model(np.array(train_x), np.array(train_y), np.array(train_weights), approach_type)
        for t in test_data:
            self.data_sizes[t] = len(test_data[t][0])
            self.results[t] = collections.OrderedDict()
            test_data_x, test_data_y, test_data_weights = test_data[t]
            if train_y.count(train_y[0]) == len(train_y): # All training examples are of the same class
                print("All datapoints have the same class: ",train_y[0])
                prediction_classes = np.zeros(len(test_data_x))
                prediction_classes[:] = train_y[0]
                prediction_probs = np.zeros([len(test_data_x),2])
                prediction_probs[:] = train_y[0]
                if not isinstance(prediction_probs[0], float):
                    prediction_probs = prediction_probs[:,1]
            else: # Use model to predict classes for test set
                prediction_probs = model.predict_proba(test_data_x)
                if not isinstance(prediction_probs[0], float):
                    prediction_probs = prediction_probs[:,1]
                prediction_classes = [1 if x >= best_threshold else 0 for x in prediction_probs]
                score = getattr(metrics, 'f1_score')(test_data_y, prediction_classes, sample_weight=test_data_weights)
            if t == "all":
                self.prediction_probs_all = {}
                self.prediction_classes_all = {}
                for j in range(len(test_data_x)):
                    self.prediction_probs_all[tuple(test_data_x[j])] = prediction_probs[j]
                    self.prediction_classes_all[tuple(test_data_x[j])] = prediction_classes[j]
            for m in all_metric_scores:
                if m in ['mean_squared_error', 'roc_auc_score', 'average_precision_score']:
                    if m == 'roc_auc_score' and sum(test_data_y) == 0: # If all true y labels are class 0, roc_auc_score cannot be calculated, so set it to 0
                        self.results[t][m] = 0
                    else:
                        if sum(test_data_weights) == 0:
                            single_weight = 1/len(test_data_weights)
                            for index_i in range(len(test_data_weights)):
                                test_data_weights[index_i] = single_weight
                        self.results[t][m] = getattr(metrics, m)(test_data_y, prediction_probs, sample_weight=test_data_weights)
                else:
                    self.results[t][m] = getattr(metrics, m)(test_data_y, prediction_classes, sample_weight=test_data_weights)
        return self.prediction_probs_all, self.prediction_classes_all

    def train_full_model(self, x, y, weights, approach_type):
        sample_weights = None
        if len(weights) > 0:
            sample_weights = weights

        candidate_params = list(ParameterSampler(param_distributions=self.params, n_iter=10, random_state=None))
        model = clone(self.classifier)
        num_splits = 3
        if sum(y) == 1: #Only one 1 in the data
            model, score = self.train_folds(model, x, y, weights, x, y)
            return model, 0.5
        elif sum(y) < num_splits:
            num_splits = sum(y)
        best_params = None
        best_score = -1
        # Loop through all possible parameter configurations and find the best one
        for parameters in candidate_params:
            model.set_params(**parameters)
            cv = StratifiedKFold(n_splits=num_splits, random_state=None, shuffle=True)
            cv_scores = []
            for train, test in cv.split(x, y):
                x_train_, x_test_, y_train_, y_test_, weights_train_, weights_test_ = x[train], x[test], y[train], y[test], weights[train], weights[test]
                tmp_model, score = self.train_folds(model, x_train_, y_train_, weights_train_, x_test_, y_test_)
                cv_scores.append(score)

            avg_score = float(sum(cv_scores))/len(cv_scores)
            if avg_score > best_score:
                best_params = parameters
                best_score = avg_score

        # Train the final model. Leave a portion of the training data out for choosing a threshold (to obtain calibrated estimates)
        final_model = clone(model)
        final_model.set_params(**best_params)
        testsize = 0.33
        if sum(y) < 3:
            testsize = 0.5
        x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(x, y, weights, test_size=testsize, shuffle=True, stratify=y)
        init_percent_positive, over_x_train, over_y_train, over_weights_train = self.oversample(x_train, y_train, weights_train)
        final_model.fit(over_x_train, over_y_train, sample_weight=over_weights_train)

        prediction_probs = final_model.predict_proba(x_test)
        if not isinstance(prediction_probs[0], float):
            prediction_probs = prediction_probs[:,1]
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, prediction_probs, pos_label=1)
        count = len(thresholds)-1
        # Choose a threshold that results in a similar percentage of predicted blind spots compared to the percentage in the data used to train the model
        while count >= 0:
            threshold = thresholds[count]
            prediction_classes = [1 if x >= threshold else 0 for x in prediction_probs]
            percent_positive = self.get_percent_positive(prediction_classes)
            if percent_positive > init_percent_positive:
                if count < len(thresholds)-1:
                    count = count+1
                threshold = thresholds[count]
                break
            count = count-1
        print("Fitted. Threshold: ",threshold)
        return final_model, threshold

    def train_folds(self, model, x_train_, y_train_, weights_train_, x_test_, y_test_):
        init_percent_positive, x_train, y_train, weights_train = self.oversample(x_train_, y_train_, weights_train_)
        _, x_test, y_test, weights_test = self.oversample(x_test_, y_test_, y_test_)
        model.fit(x_train, y_train, sample_weight=weights_train)

        prediction_classes = model.predict(x_test)
        test_weights = []
        for i in range(len(x_test)):
            test_weights.append(self.state_visits[tuple(x_test[i])])
        f1_score = getattr(metrics, 'f1_score')(y_test, prediction_classes, sample_weight=test_weights)
        return model, f1_score

    def get_percent_positive(self, arr):
        return float(sum(arr))/len(arr)

    def oversample(self, x_, y_, weights_):
        x = list(x_)
        y = list(y_)
        weights = list(weights_)
        # Oversamples positive examples (datapoints with a 1 label) until classes are balanced in the data
        percent_positive = float(sum(y))/len(y)
        init_percent_positive = percent_positive
        positive_indices = [index for index in range(len(y)) if y[index] > 0]
        while percent_positive < 0.5: # Oversample data until 1's and 0's are balanced (50/50)
            index = np.random.choice(positive_indices, 1)[0]
            y.append(y[index])
            x.append(x[index])
            weights.append(weights[index])
            percent_positive = float(sum(y))/len(y)
        shuffled_x, shuffled_y, shuffled_weights = shuffle(x, y, weights)
        return init_percent_positive, np.array(shuffled_x), np.array(shuffled_y), np.array(shuffled_weights)

def main(save_dir, training_filename, testing_filename, label, approach_type, metrics, state_visits_file):
    state_visits = review.read_labelled_file(state_visits_file)
    visit_sum = 0
    for state in state_visits:
        visit_sum += state_visits[state]
    for state in state_visits:
        state_visits[state] = state_visits[state]/float(visit_sum)
    classifier = Classifier(save_dir)
    prediction_probs, prediction_classes = classifier.train_test(training_filename, testing_filename, state_visits, label, approach_type, metrics)
    return classifier.results, classifier.data_sizes, prediction_probs, prediction_classes