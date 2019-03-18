## Discovering Blind Spots in Reinforcement Learning

### Installation:

* Clone this repository:

	```
	git clone https://github.com/ramya-ram/discovering-blind-spots.git
	cd discovering-blind-spots
	```

* Set up an Anaconda environment with the required packages:
	```
	conda env create -f environment.yml
	source activate discovering-blind-spots
	```

* Install the included domains:

	```
	cd domains
	pip install -e .
	cd ..
	```

### Obtaining policies from domains:

* The inputfiles directory contains learned Q-value files for the domains used in our experiments. If you want to obtain the optimal policies yourself for custom source and target domains:

	```
	cd run_q_learning
	python run_game.py (game-name) (save-directory-name)

	e.g.
	python run_game.py SourceCatcher-v0 source_catcher
	python run_game.py TargetCatcher-v0 target_catcher
	python run_game.py SourceFlappyBird-v0 source_flappybird
	python run_game.py TargetFlappyBird-v0 target_flappybird
	```

	The learned Q-values, mean reward learning curve, and other debug info (state counts, etc.) will be saved to the specified directory location.
	The domain code files for Catcher and FlappyBird are located in domains/domains/ple. The code for the Q-learning part is in the run_q_learning folder and includes run_game.py and q_learner.py.

* If you want to watch the agent play the learned source/target tasks:

	```
	cd run_q_learning
	python run_game.py (game-name) (save-directory-name) (learned-Q-file)

	e.g.
	python run_game.py SourceCatcher-v0 source_catcher_learned source_catcher/Q.csv
	python run_game.py TargetCatcher-v0 target_catcher_learned target_catcher/Q.csv
	python run_game.py SourceFlappyBird-v0 source_flappybird_learned source_flappybird/Q.csv
	python run_game.py TargetFlappyBird-v0 target_flappybird_learned target_flappybird/Q.csv
	```

### Running approach to identify blind spots:

* To run our approach with the learned source and target policies:

	```
	cd run_approach
	python -W ignore test_approach.py (config-file)

	e.g.
	python -W ignore test_approach.py run_config.yml
	```

	The argument "-W ignore" is used to ignore warning messages. The config file is in yaml format. An example is provided in run_approach/run_config.yml, and the config files used for experiments are in the inputfiles directory. The parameters to include in the config file are:

	* save_dir: the directory for saving results.
	* label_types: the label types you want to run as a list. The possible ones include: "corrections", "demo-action-mismatch","demo-acceptable","random-action-mismatch","random-acceptable".
	* num_runs: the number of runs that the results will be averaged over.
	* sourceQ_file: the learned source Q file.
	* targetQ_file: the learned target Q file.
	* target_env: the matching target task environment for the target Q file provided in the previous parameter.
	* percentile: percentile that determines how much action mismatch noise will be present. The possible values are -1 or a float between 0 and 1. If -1 is specified, the only acceptable actions are optimal actions (any suboptimal action is unacceptable), which results in the most strict acceptability function. A percentile of 0.95 would lead to a more lenient acceptability function (more acceptable actions) than a percentile of 0.7.
	* budget_list: list of budgets to run the approach over. This should be written as a string in the format "range(start, stop, step)", as is used in Python.
	* percent_data: percent of simulator states that are visited during data collection. If 0.8 is specified, 80% of states will be seen in training and 20% will be left out for testing.
	* state_visits: If using pre-learned files from the inputfiles directory, this has already been computed - you can pass the state_counts.csv file specifying state visit frequencies when running pi_sim on the target world. These frequencies are used to assign more importance to highly-visited states when computing F1-scores. If you trained an agent on your own custom environments and do not have this pre-computed, you can just put "", and the state visit file will be created automatically before running the algorithm.

	The code for the approach and baselines can be found in the run_approach folder, which includes test_approach.py, review.py, dawid_skene.py, classifier.py, and baselines.py.

* To obtain plots:

	```
	cd run_approach
	python plot_graphs.py (directory-with-results)

	e.g.
	python plot_graphs.py result_dir
	```

	This will create:
	* plots for assessing classifier performance, comparing our method to baseline approaches.
	* plots for oracle-in-the-loop evaluation, which compares an agent that uses our blind spot model to query an oracle for help vs. an agent that always queries vs. one that never queries.