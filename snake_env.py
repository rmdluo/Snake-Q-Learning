import random
from snake.snake_model import Snake, Action, GameResponse

class SnakeEnv:
    def __init__(self):
        self.game = Snake()
        self.action_space = [
            Action.LEFT,
            Action.RIGHT,
            Action.UP,
            Action.DOWN,
        ]

    def reset(self):
        self.game.init_game()
        return self.get_state()

    def get_state(self):
        return self.game.get_board()

    def step(self, action):
        game_response = self.game.step(action)

        # get current state
        state = self.get_state()

        # get the reward
        reward = 0
        if game_response == GameResponse.LOSE:
            reward = -1
        elif game_response == GameResponse.APPLE:
            reward = 1
        
        # get whether the game has terminated
        terminated = game_response == GameResponse.LOSE

        return state, reward, terminated
    
    def get_actions(self):
        return self.action_space
    
    def get_dummy_action(self):
        return Action.NONE

    def sample_action(self):
        return random.sample(self.action_space, 1)[0]
    
    def sample_action_no_opposite(self, last_action):
        valid_actions = [
            action for action in self.action_space if not (action == Action.get_opposite(last_action))
        ]

        if len(valid_actions) == 0:
            return random.sample(self.action_space, 1)[0]
        
        return random.sample(valid_actions, 1)[0]
    
    def sample_action_no_die_no_opposite(self, last_action):
        valid_actions = [
            action for action in self.action_space if not (self.game.is_losing(action) or action == Action.get_opposite(last_action))
        ]

        if len(valid_actions) == 0:
            return random.sample(self.action_space, 1)[0]
        
        return random.sample(valid_actions, 1)[0]
    
    def check_action(self, action, last_action):
        return action != Action.get_opposite(last_action)
    
    def __str__(self):
        return self.game.__str__()