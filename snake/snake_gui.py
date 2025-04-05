### Generated using ChatGPT
import tkinter as tk
from threading import Thread, Lock
import queue
import time

from snake_model import Snake, Action, GameResponse

class SnakeGameGUI:
    def __init__(self, cell_size=20):
        self.game = Snake()
        self.game.init_game()

        self.cell_size = cell_size
        self.width = self.game.width * cell_size
        self.height = self.game.height * cell_size

        self.window = tk.Tk()
        self.window.title("Snake Game")
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        # Input queue and lock for thread safety
        self.input_queue = queue.Queue()
        self.queue_lock = Lock()

        self.running = True
        self.current_action = Action.NONE

        self.loop_id = None

        # Bind input to keypress handler
        self.window.bind("<KeyPress>", self.on_key_press)

        # Start background thread to manage input
        self.input_thread = Thread(target=self.input_worker, daemon=True)
        self.input_thread.start()

        self.draw_game()
        self.game_loop()

        self.window.mainloop()


    def restart_game(self):
        if self.loop_id:
            self.window.after_cancel(self.loop_id)
            self.loop_id = None

        self.input_queue.queue.clear()
        self.canvas.delete("all")
        self.running = True
        self.current_action = Action.NONE
        self.game.init_game()
        self.draw_game()
        self.game_loop()


    def on_key_press(self, event):
        key = event.keysym.lower()

        if key == 'r':
            self.restart_game()
            return
        elif key == 'q':
            self.window.quit()
            return

        action = Action.NONE
        if key in ['w', 'up']:
            action = Action.UP
        elif key in ['s', 'down']:
            action = Action.DOWN
        elif key in ['a', 'left']:
            action = Action.LEFT
        elif key in ['d', 'right']:
            action = Action.RIGHT

        if action != Action.NONE:
            with self.queue_lock:
                self.input_queue.put(action)

    def input_worker(self):
        while self.running:
            # Sleep very briefly to reduce CPU usage
            time.sleep(0.01)

    def get_next_action(self):
        with self.queue_lock:
            while not self.input_queue.empty():
                next_action = self.input_queue.get()
                # Skip invalid reverse direction input
                if len(self.game.snake) <= 1 or next_action != Action.get_opposite(self.current_action):
                    self.current_action = next_action
                    break
        return self.current_action

    def draw_game(self):
        self.canvas.delete("all")
        for (x, y) in self.game.snake:
            self.draw_cell(x, y, "green")
        head_x, head_y = self.game.get_snake_head()
        self.draw_cell(head_x, head_y, "lightgreen")
        apple_x, apple_y = self.game.get_apple()
        self.draw_cell(apple_x, apple_y, "red")

    def draw_cell(self, x, y, color):
        x1 = x * self.cell_size
        y1 = y * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

    def game_loop(self):
        if self.running:
            action = self.get_next_action()
            response = self.game.step(action)
            if response == GameResponse.LOSE:
                self.running = False
                self.canvas.create_text(
                    self.width // 2,
                    self.height // 2,
                    text="Game Over",
                    fill="white",
                    font=("Arial", 24)
                )
            else:
                self.draw_game()
                self.loop_id = self.window.after(150, self.game_loop)


if __name__ == '__main__':
    SnakeGameGUI()
