from ple.games.flappybird import FlappyBird
from ple.games.flappybird import Pipe
from ple.games.flappybird import BirdPlayer
import os
import sys
import numpy as np
import pygame
from pygame.constants import K_w
from ple.games import base
import random
import collections
from itertools import product

class SourceFlappyBird(FlappyBird):
    def __init__(self, width=288, height=512, pipe_gap=100):
        super(SourceFlappyBird, self).__init__(width, height, pipe_gap)

        actions = collections.OrderedDict() # To force order of actions to be the same each time
        actions["up"] = K_w
        base.PyGameWrapper.__init__(self, width, height, actions=actions)
        self.states = []

    def init(self):
        super(SourceFlappyBird, self).init()
        self.pipe_min = 25
        self.pipe_max = 150
        self.init_pos = (
            int(self.width * 0.2),
            int(self.height / 2 - 100)
         )
        # print(self.init_pos)
        self.y_interval = 50
        self.y_max_pos = 300
        self.feature_bins = [[self.pipe_min+self.pipe_gap, self.pipe_max+self.pipe_gap],
                              [i for i in range(0, 251, 50)],
                              [self.pipe_min, self.pipe_max],
                              [i for i in range(-10, 11, 5)],
                              [i for i in range(0, self.y_max_pos+1, self.y_interval)]]
        self.feature_map = {"next_pipe_bottom_y": 0, "next_pipe_dist_to_player": 1, "next_pipe_top_y": 2,
                            "player_vel": 3, "player_y": 4}

        if len(self.states) == 0:
            for s in product(*self.feature_bins):
                # Remove states where the top and bottom pipes are not a distance of pipe_gap in between.
                if s[self.feature_map["next_pipe_bottom_y"]] != s[self.feature_map["next_pipe_top_y"]] + self.pipe_gap:
                    continue
                else:
                    self.states.append(s)
            print(len(self.states))

        self.pipe_group = pygame.sprite.Group([self._generatePipes(offset=-75)])
        for i, p in enumerate(self.pipe_group):
            self._generatePipes(offset=self.pipe_offsets[i], pipe=p)

        self.win = False

    def getGameState(self):
        # State = [player y position, player velocity, next pipe distance to player,
        #          next pipe top y position, next pipe bottom y position]
        self.pipes = []
        for p in self.pipe_group:
            self.pipes.append((p, p.x - self.player.pos_x))
        self.pipes.sort(key=lambda p: p[1])

        next_pipe = self.pipes[0][0]

        self.state_dict = {
            "player_y": self.player.pos_y,
            "player_vel": self.player.vel,

            "next_pipe_dist_to_player": next_pipe.x - self.player.pos_x,
            "next_pipe_top_y": next_pipe.gap_start,
            "next_pipe_bottom_y": next_pipe.gap_start + self.pipe_gap
        }
        for i in self.state_dict:
            possible_values = self.feature_bins[self.feature_map[i]]
            self.state_dict[i] = possible_values[np.digitize(self.state_dict[i], possible_values)-1]
        return [int(x) for x in self.get_arr_state(self.state_dict)]

    def get_arr_state(self, state):
        arr_state = np.zeros(len(state))
        i=0
        for key in sorted(state.keys()):
            index = self.feature_map[key]
            digitized_value = self.feature_bins[index][np.digitize(state[key], self.feature_bins[index])-1]
            arr_state[i] = digitized_value
            i+=1
        return arr_state

    def get_pipe_color(self, start_gap):
        return "green"

    def create_pipe(self, start_gap, offset):
        return Pipe(
                self.width,
                self.height,
                start_gap,
                self.pipe_gap,
                self.images["pipes"],
                self.scale,
                offset=offset,
                color=self.pipe_color
            )

    def _generatePipes(self, offset=0, pipe=None):
        possible_gaps = [0]
        if hasattr(self, "feature_map"):
            possible_gaps = self.feature_bins[self.feature_map["next_pipe_top_y"]]
        start_gap = random.choice(possible_gaps)

        self.pipe_color = self.get_pipe_color(start_gap)

        if pipe is None:
            return self.create_pipe(start_gap, offset)
        else:
            pipe.init(start_gap, self.pipe_gap, offset, self.pipe_color)

    def game_over(self):
        return self.lives <= 0 or self.win

    def update_reward(self):
        for p in self.pipe_group:
            if self.player.pos_y <= 50:
                self.score += 0.1

    def step(self, dt):
        self.rewards["tick"] = 0
        self.rewards["positive"] = 10
        self.rewards["loss"] = -10
        super(SourceFlappyBird, self).step(dt)
        for p in self.pipe_group:
            # If bird is under the pipe, wins this episode!
            if (p.x - p.width / 2) <= self.player.pos_x < (p.x - p.width / 2 + 4):
                self.score += self.rewards["positive"]
                self.win = True
        self.update_reward()