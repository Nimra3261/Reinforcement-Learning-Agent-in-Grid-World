import tkinter as tk
import numpy as np
import time
from PIL import Image, ImageTk

class GridWorld:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.state = (0, 0)
        self.goal_state = (grid_size-1, grid_size-1)
        self.cat_state = (2, 2)  # Position of the cat
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.grid_size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.grid_size - 1:
            y += 1

        self.state = (x, y)
        reward = -10 if self.state == self.cat_state else 10 if self.state == self.goal_state else -0.1
        done = self.state == self.goal_state or self.state == self.cat_state
        return self.state, reward, done

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        x, y = self.state
        gx, gy = self.goal_state
        cx, cy = self.cat_state
        grid[x, y] = -1  # Bee
        grid[gx, gy] = 1  # Flower
        grid[cx, cy] = -2  # Cat
        return grid

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(env.actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            x, y = state
            return self.env.actions[np.argmax(self.q_table[x, y])]

    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_idx = self.env.actions.index(action)
        best_next_action = np.argmax(self.q_table[next_x, next_y])
        td_target = reward + self.gamma * self.q_table[next_x, next_y, best_next_action]
        td_error = td_target - self.q_table[x, y, action_idx]
        self.q_table[x, y, action_idx] += self.alpha * td_error

    def train(self, num_episodes=10000):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                if done:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class RLVisualizer(tk.Tk):
    def __init__(self, agent, env):
        super().__init__()
        self.agent = agent
        self.env = env
        self.grid_size = env.grid_size
        self.cell_size = 100  # Increased cell size for better visibility
        self.title("RL Agent Visualizer")
        self.geometry(f"{self.grid_size * self.cell_size}x{self.grid_size * self.cell_size + 50}")
        self.canvas = tk.Canvas(self, width=self.grid_size * self.cell_size, height=self.grid_size * self.cell_size)
        self.canvas.pack()
        
        self.info_label = tk.Label(self, text="")
        self.info_label.pack()

        # Load and resize images
        self.bee_image = Image.open("bee.png").resize((self.cell_size, self.cell_size), Image.LANCZOS)
        self.flower_image = Image.open("flower.png").resize((self.cell_size, self.cell_size), Image.LANCZOS)
        self.cat_image = Image.open("cat.png").resize((self.cell_size, self.cell_size), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.bee_photo = ImageTk.PhotoImage(self.bee_image)
        self.flower_photo = ImageTk.PhotoImage(self.flower_image)
        self.cat_photo = ImageTk.PhotoImage(self.cat_image)
        
        self.cumulative_reward = 0
        self.after(0, self.run_episode)

    def render_grid(self):
        self.canvas.delete("all")
        grid = self.env.render()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                if grid[i, j] == -1:
                    self.canvas.create_image((x0 + self.cell_size // 2, y0 + self.cell_size // 2), image=self.bee_photo)
                elif grid[i, j] == 1:
                    self.canvas.create_image((x0 + self.cell_size // 2, y0 + self.cell_size // 2), image=self.flower_photo)
                elif grid[i, j] == -2:
                    self.canvas.create_image((x0 + self.cell_size // 2, y0 + self.cell_size // 2), image=self.cat_photo)
                else:
                    self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")

    def run_episode(self):
        state = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            self.render_grid()
            self.update_idletasks()
            self.update()
            time.sleep(0.5)
            action = self.agent.choose_action(state)
            state, reward, done = self.env.step(action)
            episode_reward += reward
            self.cumulative_reward += reward
            self.info_label.config(text=f"Current Reward: {episode_reward:.2f} | Cumulative Reward: {self.cumulative_reward:.2f}")
        self.render_grid()
        self.after(1000, self.run_episode)

if __name__ == "__main__":
    env = GridWorld(grid_size=5)
    agent = QLearningAgent(env)
    agent.train(num_episodes=10000)
    
    app = RLVisualizer(agent, env)
    app.mainloop()


