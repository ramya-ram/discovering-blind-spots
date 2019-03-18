from ple.games.catcher import Catcher
from ple.games.catcher import Paddle
from ple.games.catcher import Fruit
import pygame
import random
import collections
from pygame.constants import K_a, K_d
from ple.games import base
import numpy as np
from itertools import product

class AgentPaddle(Paddle):
    def __init__(self, speed, width, height, SCREEN_WIDTH, SCREEN_HEIGHT, feature_bins):
        super(AgentPaddle, self).__init__(speed, width, height, SCREEN_WIDTH, SCREEN_HEIGHT)
        x = random.choice(feature_bins) # Choose randomly from one of the possible x locations
        self.rect.center = (x, SCREEN_HEIGHT - height - 3)

    def update(self, dx, dt):
        x, y = self.rect.center
        n_x = x + dx

        if n_x <= 0:
            n_x = 0

        if n_x + self.width >= self.SCREEN_WIDTH:
            n_x = self.SCREEN_WIDTH - self.width

        self.rect.center = (n_x, y)

class GoodFruit(Fruit):
    def __init__(self, speed, size, SCREEN_WIDTH, SCREEN_HEIGHT, rng, y_feature_bins):
        super(GoodFruit, self).__init__(speed, size, SCREEN_WIDTH, SCREEN_HEIGHT, rng)
        self.y_feature_bins = y_feature_bins

    def update(self, dt):
        x, y = self.rect.center
        curr_y_bin = np.digitize(y, self.y_feature_bins)-1
        new_y_bin = curr_y_bin + self.speed
        if new_y_bin >= len(self.y_feature_bins):
            new_y_bin = len(self.y_feature_bins)-1 
        n_y = self.y_feature_bins[new_y_bin]

        self.rect.center = (x, n_y)

    def reset(self, x_bins):
        x = random.choice(x_bins) # Choose randomly from one of the possible values in x_bins
        y = 0
        self.rect.center = (x, y)

class SourceCatcher(Catcher):
    def __init__(self, width=500, height=500, init_lives=0, max_steps=100):
        super(SourceCatcher, self).__init__(width, height, init_lives)

        actions = collections.OrderedDict() # To force order of actions to be the same each time
        actions["left"] = K_a
        actions["right"] = K_d

        base.PyGameWrapper.__init__(self, width, height, actions=actions)
        self.player_speed = 50
        self.fruit_fall_speed = 1
        self.fruit_size = 50
        self.paddle_width = 50
        self.paddle_height = 50
        self.step_num = 0
        self.max_steps = max_steps

        self.feature_bins = [range(0, self.width, self.player_speed),
                             range(0, self.width, self.fruit_size),
                             range(0, self.height, self.fruit_size)]
        self.feature_map = {"player_x":0, "fruit_x":1, "fruit_y": 2}

        self.y_locs = self.feature_bins[self.feature_map["fruit_y"]]
        self.states = []
        for s in product(*self.feature_bins):
            # Remove states in which the fruit has already hit the ground (The agent does not take an action in these states because they are goal states - Q-value will be 0)
            if s[self.feature_map["fruit_y"]] == self.y_locs[len(self.y_locs)-1]:
                continue
            else:
                self.states.append(s)
        print(len(self.states))

    def init(self):
        self.score = 0
        self.player = AgentPaddle(self.player_speed, self.paddle_width,
                             self.paddle_height, self.width, self.height, self.feature_bins[self.feature_map["player_x"]])

        self.fruit = GoodFruit(self.fruit_fall_speed, self.fruit_size,
                           self.width, self.height, self.rng, self.feature_bins[self.feature_map["fruit_y"]])

        self.fruit.reset(self.feature_bins[self.feature_map["fruit_x"]])
        self.step_num = 0
        self.ended = False
        # self.player.draw(self.screen)
        # self.fruit.draw(self.screen)

    def getGameState(self):
        # State = [player x position, fruit x position, fruit y position]
        state = [self.player.rect.center[0], self.fruit.rect.center[0], self.fruit.rect.center[1]]
        return state

    def game_over(self):
        return self.ended

    def step(self, dt):
        self.step_num += 1
        self.screen.fill((0, 0, 0))
        self._handle_player_events()

        self.curr_score = 0

        self.player.update(self.dx, dt)
        self.fruit.update(dt)

        self.player.draw(self.screen)
        self.fruit.draw(self.screen)

        self.update_fruit_score()
        self.score += self.curr_score

    def update_fruit_score(self):
        max_dist = self.width - self.paddle_width
        dist = abs(self.player.rect.center[0] - self.fruit.rect.center[0])
        self.curr_score = ((max_dist-dist)/self.player_speed)
        y_feature_bins = self.feature_bins[self.feature_map["fruit_y"]]
        if self.fruit.rect.center[1] == y_feature_bins[len(y_feature_bins)-1]: # Last y-value
            self.ended = True