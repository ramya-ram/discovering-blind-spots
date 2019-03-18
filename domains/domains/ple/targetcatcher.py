from domains.ple.sourcecatcher import SourceCatcher
from domains.ple.sourcecatcher import GoodFruit
from domains.ple.sourcecatcher import AgentPaddle
import random
import pygame
import numpy as np
import math
from itertools import product

class UncertainFruit(GoodFruit):
    def __init__(self, speed, size, SCREEN_WIDTH, SCREEN_HEIGHT, rng, y_feature_bins, fruit_num_types, blindspot_region, init_probGoodFruit):
        super(UncertainFruit, self).__init__(speed, size, SCREEN_WIDTH, SCREEN_HEIGHT, rng, y_feature_bins)
        self.num_types = fruit_num_types
        self.blindspot_region = blindspot_region
        self.init_probGoodFruit = init_probGoodFruit
        self.type = -1 # Initialization of type

    def drawFruit(self, fruitType, bad_fruit_id):
        if fruitType == bad_fruit_id:
            color = (50, 50, 50)
        else:
            color = (255, 120, 120)
        image = pygame.Surface((self.size, self.size))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            color,
            (0, 0, self.size, self.size),
            0
        )
        self.image = image

    def reset(self, x_bins, bad_fruit_id):
        super(UncertainFruit, self).reset(x_bins)
        self.setFruitType(self.rect.center[0], bad_fruit_id)
        self.drawFruit(self.type, bad_fruit_id)

    def setFruitType(self, fruit_x, bad_fruit_id):
        # For fruit_x values between 0 and the threshold, the target world looks exactly like the source (only good fruit).
        # For x values after the threshold, good fruit falls with probability probGoodFruit and bad fruit falls with probability (1-probGoodFruit)
        if fruit_x in self.blindspot_region:
            self.probGoodFruit = self.init_probGoodFruit
        else:
            self.probGoodFruit = 1
        if random.random() < self.probGoodFruit:
            self.type = random.randint(1, self.num_types-1) # Good fruit
        else:
            self.type = bad_fruit_id # Bad fruit

class TargetCatcher(SourceCatcher):
    def __init__(self, width=500, height=500, init_lives=0, max_steps=100):
        super(TargetCatcher, self).__init__(width, height, init_lives)
        self.bad_fruit_id = 0 # Bad fruits have type 0. All other fruits have type > 0.
        self.fruit_num_types = 2
        self.init_probGoodFruit = 0.5
        self.feature_bins = [range(0, self.width, self.player_speed),
                             range(0, self.width, self.fruit_size),
                             range(0, self.height, self.fruit_size),
                             range(0, self.fruit_num_types, 1)]
        self.feature_map = {"player_x":0, "fruit_x":1, "fruit_y": 2, "fruit_type":3}
        self.states = []

    def init(self):
        super(TargetCatcher, self).init()
        self.blindspot_region = [250,300,350,400,450]

        if len(self.blindspot_region) > 0 and len(self.states) == 0:
            for s in product(*self.feature_bins):
                # Remove states in which the fruit has already hit the ground (The agent does not take an action in these states because they are goal states - Q-value will be 0).
                if s[self.feature_map["fruit_y"]] == self.y_locs[len(self.y_locs)-1]:
                    continue
                # Remove states with bad fruits in the "good" region. There will never be a bad fruit initialized there.
                elif s[self.feature_map["fruit_x"]] not in self.blindspot_region and s[self.feature_map["fruit_type"]] == self.bad_fruit_id:
                    continue
                else:
                    self.states.append(s)
            print(len(self.states))

        self.fruit = UncertainFruit(self.fruit_fall_speed, self.fruit_size,
                           self.width, self.height, self.rng,
                           self.feature_bins[self.feature_map["fruit_y"]], self.fruit_num_types, self.blindspot_region, self.init_probGoodFruit)

        if not hasattr(self, "locs"):
            self.locs = self.feature_bins[self.feature_map["fruit_x"]]
        self.fruit.reset(self.locs, self.bad_fruit_id)
        self.real_state = True
        # self.player.draw(self.screen)
        # self.fruit.draw(self.screen)

    def getGameState(self):
        # State = [player x position, fruit x position, fruit y position, fruit type]
        state = [self.player.rect.center[0], self.fruit.rect.center[0], self.fruit.rect.center[1]]
        if self.real_state:
            state = [self.player.rect.center[0], self.fruit.rect.center[0], self.fruit.rect.center[1], self.fruit.type]
        return state

    def get_source_state(self, state):
        return state[:-1]

    def update_fruit_score(self):
        if self.fruit.type == self.bad_fruit_id:
            dist = abs(self.player.rect.center[0] - self.fruit.rect.center[0])
            self.curr_score = (dist/self.player_speed)
            if dist == 0:
                self.curr_score = -100
            y_feature_bins = self.feature_bins[self.feature_map["fruit_y"]]
            if self.fruit.rect.center[1] == y_feature_bins[len(y_feature_bins)-1]: # Last y-value
                self.ended = True
        else: # If good fruit (exactly like source)
            super(TargetCatcher, self).update_fruit_score()

    def state_in_locs(self, state):
        return state[self.feature_map["fruit_x"]] in self.locs

    def generate_training_subset(self, percent_sim_data):
        # Choose a subset of initial states, based on what percentage of data the agent should see in training
        self.all_locs = self.feature_bins[self.feature_map["fruit_x"]]
        half_mark = int(len(self.all_locs)/2)
        all_safe_locs = self.all_locs[0:half_mark]
        all_bs_locs = self.all_locs[half_mark:len(self.all_locs)]
        num_initstates_from_each = int(len(self.all_locs)*(percent_sim_data/2))
        safe_locs = random.sample(all_safe_locs, num_initstates_from_each)
        bs_locs = random.sample(all_bs_locs, num_initstates_from_each)
        self.training_locs = safe_locs + bs_locs

    def set_to_training_set(self):
        self.locs = self.training_locs

    def set_to_testing_set(self):
        self.locs = self.all_locs

    def get_uniform_state_weights(self):
        weights = {}
        for s in self.states:
            if s[self.feature_map["fruit_type"]] == self.bad_fruit_id: # Blind spot
                weights[s] = self.init_probGoodFruit
                good_state = []
                for i in range(len(s)-1):
                    good_state.append(s[i])
                good_state.append(1)
                good_state = tuple(good_state)
                weights[good_state] = 1-self.init_probGoodFruit
            elif s not in weights:
                weights[s] = 1

        states_list = []
        weights_list = []
        for s in self.states:
            states_list.append(s)
            weights_list.append(weights[s])
        weights_list = [float(i)/sum(weights_list) for i in weights_list]
        return states_list, weights_list