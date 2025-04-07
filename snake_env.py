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
        return [
            int(self.game.is_losing(Action.LEFT)), # does going left lose
            int(self.game.is_losing(Action.RIGHT)), # does going right lose
            int(self.game.is_losing(Action.UP)), # does going up lose
            int(self.game.is_losing(Action.DOWN)), # does going down lose
            int(self.game.get_last_action() == Action.LEFT), # is the snake going left
            int(self.game.get_last_action() == Action.RIGHT), # is the snake going right
            int(self.game.get_last_action() == Action.UP), # is the snake going up
            int(self.game.get_last_action() == Action.DOWN), # is the snake going down
            int(self.game.get_snake_head()[0] > self.game.get_apple()[0]), # is the apple to the left of the snake
            int(self.game.get_snake_head()[0] < self.game.get_apple()[0]), # is the apple to the right of the snake
            int(self.game.get_snake_head()[1] > self.game.get_apple()[1]), # is the apple above the snake
            int(self.game.get_snake_head()[1] < self.game.get_apple()[1]), # is the apple below the snake
        ]

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
    
    def sample_action_with_checks(self, last_action):
        action = random.sample(self.action_space, 1)[0]
        while action == Action.get_opposite(last_action):
            action = random.sample(self.action_space, 1)[0]
        return action
    
    def __str__(self):
        return self.game.__str__()