import random
import pygame
import numpy as np
from domains.ple.sourceflappybird import SourceFlappyBird
from domains.ple.sourceflappybird import Pipe
from itertools import product

class ColoredPipe(Pipe):
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, gap_start, gap_size, image_assets, scale, offset=0, color="green"):
        super(ColoredPipe, self).__init__(SCREEN_WIDTH, SCREEN_HEIGHT, gap_start, gap_size, image_assets, scale, offset, color)
        self.init(gap_start, gap_size, offset, color)

    def init(self, gap_start, gap_size, offset, color):
        super(ColoredPipe, self).init(gap_start, gap_size, offset, color)
        self.color_type = color

class TargetFlappyBird(SourceFlappyBird):
    def init(self):
        super(TargetFlappyBird, self).init()
        self.bad_pipe_id = 0
        self.pipe_num_types = 2
        self.prob_bad_pipe = 0.5
        self.feature_bins.append(range(0, self.pipe_num_types, 1))
        self.feature_map["type_color"] = len(self.feature_map)

        self.states = []
        for s in product(*self.feature_bins):
            # Remove states where the top and bottom pipes are not a distance of pipe_gap in between.
            if s[self.feature_map["next_pipe_bottom_y"]] != s[self.feature_map["next_pipe_top_y"]] + self.pipe_gap:
                continue
            # Remove states with copper/red pipes in the "good" region. There will never be copper/red pipes initialized there.
            elif not self.is_bad_region(s[self.feature_map["next_pipe_top_y"]]) and s[self.feature_map["type_color"]] == self.bad_pipe_id:
                continue
            else:
                self.states.append(s)

    def create_pipe(self, start_gap, offset):
        return ColoredPipe(
                self.width,
                self.height,
                start_gap,
                self.pipe_gap,
                self.images["pipes"],
                self.scale,
                offset=offset,
                color=self.pipe_color
            )

    def getGameState(self):
        # State = [player y position, player velocity, next pipe distance to player,
        #          next pipe top y position, next pipe bottom y position, next pipe color]
        arr_state = super(TargetFlappyBird, self).getGameState()
        color_type_feature = -1
        next_pipe = self.pipes[0][0]
        if next_pipe.color_type == "green":
            color_type_feature = random.randint(1, self.pipe_num_types-1)
        elif next_pipe.color_type == "red":
            color_type_feature = self.bad_pipe_id
        self.state_dict["type_color"] = color_type_feature
        return [int(x) for x in self.get_arr_state(self.state_dict)]

    def get_source_state(self, state):
        return state[:-1]

    def is_bad_region(self, start_gap):
        return start_gap == self.pipe_max

    def get_pipe_color(self, start_gap):
        if hasattr(self, "prob_bad_pipe"):
            if self.is_bad_region(start_gap):
                prob = random.uniform(0, 1)
                if prob <= self.prob_bad_pipe:
                    return "red"
                else:
                    return "green"
            else: # "Good region"
                return "green"
        return "green"

    def update_reward(self):
        for p in self.pipe_group:
            if p.color_type == "red":
                if self.player.pos_y >= 200:
                    self.score += 0.1
                if self.player.pos_y <= 50:
                    self.score += -100
            else:
                if self.player.pos_y <= 50:
                    self.score += 0.1

    def get_uniform_state_weights(self):
        weights = {}
        for s in self.states:
            if s[self.feature_map["type_color"]] == self.bad_pipe_id: # Blind spot
                weights[s] = self.prob_bad_pipe
                good_state = []
                for i in range(len(s)-1):
                    good_state.append(s[i])
                good_state.append(1)
                good_state = tuple(good_state)
                weights[good_state] = 1-self.prob_bad_pipe
            elif s not in weights:
                weights[s] = 1

        states_list = []
        weights_list = []
        for s in self.states:
            states_list.append(s)
            weights_list.append(weights[s])
        weights_list = [float(i)/sum(weights_list) for i in weights_list]
        return states_list, weights_list