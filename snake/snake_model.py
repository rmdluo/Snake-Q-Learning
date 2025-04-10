import random
from enum import Enum


class Action(Enum):
    NONE = -1
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    @staticmethod
    def get_opposite(action):
        action_opposite = {
            Action.NONE: Action.NONE,
            Action.LEFT: Action.RIGHT,
            Action.RIGHT: Action.LEFT,
            Action.UP: Action.DOWN,
            Action.DOWN: Action.UP,
        }
        return action_opposite[action]

class GameResponse(Enum):
    LOSE = -1
    NONE = 0
    APPLE = 1


class Snake:
    def __init__(self, width=20, height=15):
        # board properties
        self.width = width
        self.height = height

        # snake properties
        self.snake = []

        # apple properties
        self.apple = (width - 1, height - 1)

        # last movement input made
        self.last_action = Action.NONE

    def init_game(self):
        # initialize the snake position
        snake_head_x = random.randint(0, self.width - 1)
        snake_head_y = random.randint(0, self.height - 1)
        self.snake = [(snake_head_x, snake_head_y)]

        # initialize the apple position
        self.generate_apple()

        # initialize status trackers to defaults
        self.last_action = Action.NONE
    
    def get_score(self):
        return len(self.snake) - 1

    def generate_apple(self):
        apple_x = random.randint(0, self.width - 1)
        apple_y = random.randint(0, self.height - 1)
        while (apple_x, apple_y) in self.snake:
            apple_x = random.randint(0, self.width - 1)
            apple_y = random.randint(0, self.height - 1)
        self.apple = (apple_x, apple_y)

    def is_out_of_bounds(self, x, y):
        return 0 > x or self.width == x or 0 > y or self.height == y
    
    def get_snake_head(self):
        return self.snake[-1]
    
    def add_snake_head(self, x, y):
        self.snake.append((x, y))

    def remove_snake_tail(self):
        self.snake.pop(0)

    def get_apple(self):
        return self.apple
    
    def set_apple(self, x, y):
        self.apple = (x, y)

    def get_last_action(self):
        return self.last_action
    
    def get_board(self):
        board = [[0] * self.width for _ in range(self.height)]
        snake_set = set(self.snake)
        snake_head = self.get_snake_head()

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) == snake_head:
                    board[y][x] = 2
                elif (x, y) == self.apple:
                    board[y][x] = 3
                elif (x, y) in snake_set:
                    board[y][x] = 1
        return board

    def is_losing(self, action: Action):
        snake_head_x, snake_head_y = self.get_snake_head()
        return (
            (
                action == Action.LEFT
                and (
                    snake_head_x == 0
                    or ((snake_head_x - 1, snake_head_y) in self.snake  and self.snake[0] != (snake_head_x - 1, snake_head_y))
                )
            )
            or (
                action == Action.RIGHT
                and (
                    snake_head_x == self.width - 1
                    or ((snake_head_x + 1, snake_head_y) in self.snake  and self.snake[0] != (snake_head_x + 1, snake_head_y))
                )
            )
            or (
                action == Action.UP
                and (
                    snake_head_y == 0
                    or ((snake_head_x, snake_head_y - 1) in self.snake  and self.snake[0] != (snake_head_x, snake_head_y - 1))
                )
            )
            or (
                action == Action.DOWN
                and (
                    snake_head_y == self.height - 1
                    or ((snake_head_x, snake_head_y + 1) in self.snake and self.snake[0] != (snake_head_x, snake_head_y + 1))
                )
            )
        )

    def move_snake_head(self, action: Action):
        x_change = 0
        y_change = 0

        if action == Action.LEFT:
            x_change = -1
        elif action == Action.RIGHT:
            x_change = 1
        elif action == Action.UP:
            y_change = -1
        elif action == Action.DOWN:
            y_change = 1

        prev_x, prev_y = self.get_snake_head()
        self.add_snake_head(prev_x + x_change, prev_y + y_change)

        return self.get_snake_head() == self.get_apple()

    def step(self, action: Action):
        if action == Action.NONE and self.last_action == Action.NONE:
            return GameResponse.NONE

        if action == Action.NONE or (len(self.snake) > 1 and Action.get_opposite(action) == self.last_action):
            action = self.last_action
        self.last_action = action

        # check if snake runs into boundaries or snake
        if self.is_losing(action):
            return GameResponse.LOSE

        # move the snake
        was_apple_flag = self.move_snake_head(action)

        if not was_apple_flag:
            self.remove_snake_tail()
            return GameResponse.NONE
        else:
            self.generate_apple()
            return GameResponse.APPLE
        
    def __str__(self):
        display = ""
        snake_set = set(self.snake)
        snake_head = self.get_snake_head()

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) == snake_head:
                    display += "H"
                elif (x, y) == self.apple:
                    display += "A"
                elif (x, y) in snake_set:
                    display += "S"
                else:
                    display += "."
            display += "\n"
        return display


if __name__=='__main__':
    game = Snake()
    game.init_game()

    print(game)
    action = input("Enter up (u), down (d), left (l), or right (r): ")
    while True:
        if action.lower() == "up" or action.lower() == "u":
            action = Action.UP
            break
        elif action.lower() == "down" or action.lower() == "d":
            action = Action.DOWN
            break
        elif action.lower() == "left" or action.lower() == "l":
            action = Action.LEFT
            break
        elif action.lower() == "right" or action.lower() == "r":
            action = Action.RIGHT
            break
        else:
            action = input("Enter a valid action: ")

    while game.step(action) != GameResponse.LOSE:
        print(game)
        action = input("Enter up (u), down (d), left (l), or right (r): ")
        while True:
            if action.lower() == "up" or action.lower() == "u":
                action = Action.UP
                break
            elif action.lower() == "down" or action.lower() == "d":
                action = Action.DOWN
                break
            elif action.lower() == "left" or action.lower() == "l":
                action = Action.LEFT
                break
            elif action.lower() == "right" or action.lower() == "r":
                action = Action.RIGHT
                break
            else:
                action = input("Enter a valid action: ")
    print("You lost")
    print(game)

