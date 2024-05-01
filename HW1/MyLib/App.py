import tkinter as tk
import matplotlib.pyplot as plt
import os
import numpy as np
import random

from MyLib.simple_playground import Playground
from MyLib.QModel import Qmodel

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


# Global Variable
PLAYGROUND_ROOT_PTAH = ".\\playground\\"
INIT_PLAYGROUND = "軌道座標點.txt"
ROOT_PATH = os.path.dirname(os.path.abspath(__name__))

# UI Variable
FIGURE_SIZE = 10

# Q-Learning Variable
N_EPISODES = 5000
ALHPA = 0.1
GAMMA = 0.9

# Sensor Variable
CENTER_N_STATES = 30
COMBINE_N_STATES = 60
MAX_CENTER_DISTANCE = 100
MIN_CENTER_DISTANCE = 0
MAX_COMBINE_DISTANCE = 100
MIN_COMBINE_DISTANCE = -100
TURN_LEFT = -40
TURN_RIGHT = 40


class App():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Training Q-Learning Network")

        # Config
        self.config = dict(
            n_episodes = N_EPISODES,
            alpha = ALHPA,
            gamma = GAMMA,
        )

        self.dataset_path = ROOT_PATH + "\\datasets\\train4dAll.txt"
        self.playground_path = ROOT_PATH + "\\playground\\軌道座標點.txt"
        self.track_path = ROOT_PATH + "\\track\\track4D.txt"

        self.model = Qmodel(n_actions=80, n_states=(CENTER_N_STATES, COMBINE_N_STATES), alpha=0.1, gamma=0.9)
        self.action_list = []
        self.playground = Playground(self.playground_path)
        self.animation = None 

        # UI
        self.msg = tk.Label(self.root, text="Q-Learning", font=('Arial', 16))
        self.msg.grid(row=FIGURE_SIZE+4, column=0, columnspan=FIGURE_SIZE, rowspan=FIGURE_SIZE)


        self.init_plot_pannel()
        self.init_control_panel()
        self.init_config_panel()

    def init_plot_pannel(self):
        # Create a Matplotlib figure and axis
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 35)
        self.ax.set_ylim(-10, 55)
        self.ax.set_aspect('equal')
        self.ax.set_title('Playground')

        # Create a Tkinter canvas to embed the Matplotlib plot
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, columnspan=FIGURE_SIZE, rowspan=FIGURE_SIZE)

        # Draw playground
        self.draw_playground()

    def init_control_panel(self):
        group3 = tk.LabelFrame(self.root, text="Control", padx=5, pady=5)
        group3.grid(row=1, column=FIGURE_SIZE, pady=5)

        # Button to start some process
        self.start_button = tk.Button(group3, text="Train Model", command=self.startBtn_onclick, bg="green")
        self.start_button.grid(row=1, column=0, pady=5)
        # reset model
        self.reset_button = tk.Button(group3, text="Reset Model", command=self.resetBtn_onclick, bg="light grey")
        self.reset_button.grid(row=2, column=0, pady=5)
        # Save button
        self.save_button = tk.Button(group3, text="Save Car Path", command=self.saveBtn_onclick, bg="grey")
        self.save_button.grid(row=3, column=0, pady=5)


    def resetBtn_onclick(self):
        self.model = Qmodel(n_actions=80, n_states=(CENTER_N_STATES, COMBINE_N_STATES), alpha=0.1, gamma=0.9)
        self.playground = Playground(self.playground_path)
        self.draw_playground()
        self.animation.event_source.stop()        

    def draw_playground(self):
        p = self.playground
        playground_edge_lines = [[p.lines[i].p1.x, p.lines[i].p1.y, p.lines[i].p2.x, p.lines[i].p2.y] for i in range(len(p.lines))]

        # Draw playground
        for line in playground_edge_lines:
            self.ax.plot([line[0], line[2]], [line[1], line[3]], 'b-')

        # Draw destination
        rect = plt.Rectangle([p.destination_line.p1.x, p.destination_line.p1.y], p.destination_line.length, 1, color='g')
        self.ax.add_patch(rect)

        # Redraw the canvas
        self.canvas.draw()

    def init_config_panel(self):
        group = tk.LabelFrame(self.root, text="Model Config", padx=5, pady=5)
        group.grid(row=0, column=FIGURE_SIZE, pady=5)
        # Entry for epochs
        label = tk.Label(group, text="n_eposido:")
        label.grid(row=0, column=0, pady=5)
        self.epoch_entry = tk.Entry(group)
        self.epoch_entry.insert(0, self.config["n_episodes"])
        self.epoch_entry.grid(row=0, column=1, pady=5)
        # Entry for learning rate
        label = tk.Label(group, text="Alpha:")
        label.grid(row=1, column=0, pady=5)
        self.lr_entry = tk.Entry(group)
        self.lr_entry.insert(0, self.config["alpha"])
        self.lr_entry.grid(row=1, column=1, pady=5)
        # Entry for train val split
        label = tk.Label(group, text="Gamma:")
        label.grid(row=2, column=0, pady=5)
        self.train_val_split_entry = tk.Entry(group)
        self.train_val_split_entry.insert(0, self.config["gamma"])
        self.train_val_split_entry.grid(row=2, column=1, pady=5)

    def get_configs(self) -> dict:
        return dict(
            n_episodes = int(self.epoch_entry.get()),
            alpha = float(self.lr_entry.get()),
            gamma = float(self.train_val_split_entry.get())
        )

    def startBtn_onclick(self):
        
        def state_calucate(center_distance, left_distance, right_distance):
            center_state = int((center_distance - MIN_CENTER_DISTANCE) / (MAX_CENTER_DISTANCE - MIN_CENTER_DISTANCE) * CENTER_N_STATES)
            combine_distance = left_distance - right_distance
            combine_state = int((combine_distance - MIN_COMBINE_DISTANCE) / (MAX_COMBINE_DISTANCE - MIN_COMBINE_DISTANCE) * COMBINE_N_STATES)
            return center_state, combine_state

        def exploration_rate(n_episodes, min_rate=0.1, max_rate=1.0, decay_rate=0.01) -> float:
            return min_rate + (max_rate - min_rate) * np.exp(-decay_rate * n_episodes)


        def start_training():
            config = self.get_configs()
            self.config = config

            self.start_button.config(text="Training...", bg="grey", state="disabled")
            self.msg.config(text="Training...", fg="black")
            # try:
            p = self.playground

            for e in range(config["n_episodes"]):
                sensor_output = p.reset()
                center_distance, left_distance, right_distance = sensor_output
                center_state, combine_state = state_calucate(center_distance, left_distance, right_distance)
                self.action_list = []
                while not p.done:
                    print(f"pre_state: {center_state}, {combine_state}")

                    if np.random.random() < exploration_rate(e): # explore
                        action = random.randint(0, self.model.n_actions-1)
                        print(f"random action: {action}")
                    else:
                        action = self.model.predict((center_state, combine_state))
                        print(f"action: {action}")

                    # environment
                    next_sensor_output, reward = p.step(action)
                    next_center_distance, next_left_distance, next_right_distance = next_sensor_output
                    car_pos = [p.car.getPosition("center").x, p.car.getPosition("center").y]
                    
                    next_center_state, next_combine_state = state_calucate(next_center_distance, next_left_distance, next_right_distance)
                    self.model.update_table((center_state, combine_state), action, reward, (next_center_state, next_combine_state))
                    center_state, combine_state = next_center_state, next_combine_state
                    self.action_list.append({
                        "previous_car_pos": car_pos,
                        "pre_state": [next_center_distance, next_left_distance, next_right_distance],
                        "degree": action
                    })
                    
                if p.isAtDestination:
                    print("success")
                    break
                else:
                    print("fail")
                    p.reset()
                
            self.draw_run()
            print("Training Ended")

            # except Exception as e:
            #     self.msg.config(text=str(e), fg="red")

            self.start_button.config(text="Start Training", bg="green", state="normal")
            self.msg.config(text="Training Ended", fg="black")

        start_training()

    def saveBtn_onclick(self):
        save_path = os.path.join(ROOT_PATH, "car_path.txt")
        with open("car_path.txt", "w") as f:
            for action in self.action_list:
                f.write(f"{action['previous_car_pos']}\n")
        print("Car path saved to", save_path)
        self.msg.config(text="Car path saved to .\\car_path.txt", fg="black")


    def run(self) -> None:
        def on_closing():
            print("Closing...")
            self.root.destroy()
            exit()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()


    def draw_run(self) -> None:
        line, = self.ax.plot([], [], lw=2)
        circle = plt.Circle(self.action_list[0]["previous_car_pos"], radius=self.playground.car.radius/2, color='r')
        annotation = self.ax.annotate("", xy=(30, -10), xytext=(5, 5),
                                        textcoords="offset points", ha='right', va='bottom',
                                        bbox=dict(boxstyle='round,pad=0.3', alpha=0.7))

        def update(frame):
            xdata = [x["previous_car_pos"][0] for x in self.action_list[:frame]]
            ydata = [x["previous_car_pos"][1] for x in self.action_list[:frame]]

            line.set_data(xdata, ydata)

            circle_x, circle_y = self.action_list[frame]["previous_car_pos"][0], self.action_list[frame]["previous_car_pos"][1]
            right = self.action_list[frame]["pre_state"][1]
            left = self.action_list[frame]["pre_state"][2]
            front = self.action_list[frame]["pre_state"][0]

            circle.set_center((circle_x, circle_y))

            label = f'car pos: ({circle_x:.2f}, {circle_y:.2f})\nfront sensor: {front:.2f}\nright sensor: {right:.2f}\nleft sensor: {left:.2f}'
            annotation.set_text(label)

            self.canvas.draw()

            return line, circle, annotation

        def init():
            line.set_data([], [])
            circle.set_center((self.action_list[0]["previous_car_pos"][0], self.action_list[0]["previous_car_pos"][1]))
            self.ax.add_patch(circle)

            annotation = self.ax.annotate("", xy=(30, 0), xytext=(5, 5))
            return line, circle, annotation

        if self.animation:
            self.animation.event_source.stop()

        try:
            self.animation = FuncAnimation(self.figure, update, frames=len(self.action_list), init_func=init, interval=100, blit=True)
        except Exception as e:
            print("Fail to draw animation.")
            print(e)
            pass
