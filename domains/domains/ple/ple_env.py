import gym
from gym import spaces
from ple import PLE
import numpy as np

def process_state_prespecified(state):
    return np.array([ state.values() ])

def process_state(state):
    return np.array(state)

class PLEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, prespecified_game=True, game_name='FlappyBird', display_screen=True, rgb_state=False):
        # Open up a game state to communicate with emulator
        import importlib
        if prespecified_game:
            game_module_name = ('ple.games.%s' % game_name).lower()
        else:
            game_module_name = ('domains.ple.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        self.game = getattr(game_module, game_name)()
        self.rgb_state = rgb_state
        if self.rgb_state:
            self.game_state = PLE(self.game, fps=30, display_screen=display_screen)
        else:
            if prespecified_game:
                self.game_state = PLE(self.game, fps=30, display_screen=display_screen, state_preprocessor=process_state_prespecified)
            else:
                self.game_state = PLE(self.game, fps=30, display_screen=display_screen, state_preprocessor=process_state)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        if self.rgb_state:
            self.state_width, self.state_height = self.game_state.getScreenDims()
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.state_width, self.state_height, 3))
        else:
            self.state_dim = self.game_state.getGameStateDims()
            self.observation_space = spaces.Box(low=0, high=255, shape=self.state_dim)
        self.viewer = None
        self.feature_bins = []
        if hasattr(self.game, 'feature_bins'):
            self.feature_bins = self.game.feature_bins

    def get_source_state(self, state):
        if hasattr(self.game, 'get_source_state'):
            return self.game.get_source_state(state)
        return None

    def generate_training_subset(self, percent_sim_data):
        if hasattr(self.game, 'generate_training_subset'):
            return self.game.generate_training_subset(percent_sim_data)

    def set_to_training_set(self):
        if hasattr(self.game, 'set_to_training_set'):
            return self.game.set_to_training_set()

    def set_to_testing_set(self):
        if hasattr(self.game, 'set_to_testing_set'):
            return self.game.set_to_testing_set()

    def get_uniform_state_weights(self):
        if hasattr(self.game, 'get_uniform_state_weights'):
            return self.game.get_uniform_state_weights()
        else:
            states = self.get_states()
            weights = np.ones(len(states))
            weights = [float(i)/sum(weights) for i in weights]
            return states, weights

    def get_states(self):
        if hasattr(self.game, 'states'):
            return self.game.states

    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        state = self._get_state()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    def _get_image(self, game_state):
        image_rotated = np.fliplr(np.rot90(game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated

    def _get_state(self):
        if self.rgb_state:
            return self._get_image(self.game_state)
        else:
            return self.game_state.getGameState()

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _reset(self):
        if self.rgb_state:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.state_width, self.state_height, 3))
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=self.state_dim)
        self.game_state.reset_game()
        state = self._get_state()
        return state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image(self.game_state)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def _seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

        self.game_state.init()