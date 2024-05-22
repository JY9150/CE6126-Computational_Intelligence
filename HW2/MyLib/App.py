import os

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

from MyLib.simple_playground import Playground
from MyLib.Fuzzy import FuzzySystem, FuzzySet, FuzzyRule

import matplotlib.font_manager

# Global Variable
PLAYGROUND_ROOT_PTAH = ".\\playground\\"
INIT_PLAYGROUND = "軌道座標點.txt"
ROOT_PATH = os.path.dirname(os.path.abspath(__name__))

# UI Variable
FIGURE_SIZE = 10

# Sensor Variable
MAX_CENTER_DISTANCE = 20
MIN_CENTER_DISTANCE = 0
MAX_COMBINE_DISTANCE = 10
MIN_COMBINE_DISTANCE = -10
TURN_LEFT = -40
TURN_RIGHT = 40

# Fuzzy Variable
NUMBER_OF_POINTS_IN_FUZZY_SET = 1000


class App():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fuzzy System")

        self.playground_path = ROOT_PATH + "\\playground\\軌道座標點.txt"
        self.track_path = ROOT_PATH + "\\track\\track4D.txt"


        self.fuzzy_rules = self.get_fuzzy_rules()

        self.fuzzy = FuzzySystem(self.fuzzy_rules)
        self.playground = Playground(self.playground_path)
        self.animation = None 

        # UI
        self.msg = tk.Label(self.root, text="Fuzzy System", font=('Arial', 16))
        self.msg.grid(row=FIGURE_SIZE+4, column=0, columnspan=FIGURE_SIZE, rowspan=FIGURE_SIZE)


        self.init_plot_pannel()
        self.init_control_panel()

    def get_fuzzy_rules(self):
        triangular = lambda x, a, b, c: 0 if x <= a else (x-a)/(b-a) if a <= x <= b else (c-x)/(c-b) if b <= x <= c else 0
        down_up = lambda x, a, b: 0 if x <= a else (x-a)/(b-a) if a <= x <= b else 1 
        up_down = lambda x, a, b: 1 if x <= a else (b-x)/(b-a) if a <= x <= b else 0
        trapezoid = lambda x, a, b, c, d: 0 if x <= a else (x-a)/(b-a) if a <= x <= b else 1 if b <= x <= c else (d-x)/(d-c) if c <= x <= d else 0

        m_neg_set = FuzzySet(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, 
                        [up_down(x, 4, 8) for x in np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                        name="Near", id="m_neg_set")
        m_mid_set = FuzzySet(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, 
                            [triangular(x, 4, 8, 12) for x in np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                            name="Middle", id="m_mid_set")
        m_pos_set = FuzzySet(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, 
                            [down_up(x, 8, 12) for x in np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                            name="Far", id="m_pos_set")
        
        lr_neg_set = FuzzySet(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, 
                            [up_down(x, -3, 0) for x in np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                            name="Near Left", id="lr_neg_set")
        lr_mid_set = FuzzySet(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, 
                            [trapezoid(x, -3, -1, 1, 3) for x in np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                            name="Middle", id="lr_mid_set")
        lr_pos_set = FuzzySet(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, 
                            [down_up(x, 0, 3) for x in np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                            name="Near Right", id="lr_pos_set")

        angle_neg_set = FuzzySet(TURN_LEFT, TURN_RIGHT, 
                                [up_down(x, -40, 0) for x in np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                                name="Turn Left", id="angle_neg_set")
        angle_mid_set = FuzzySet(TURN_LEFT, TURN_RIGHT, 
                                [triangular(x, -20, 0, 20) for x in np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                                name="Straight", id="angle_mid_set")
        angle_pos_set = FuzzySet(TURN_LEFT, TURN_RIGHT, 
                                [down_up(x, 0, 40) for x in np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                                name="Turn Right", id="angle_pos_set")

        fuzzy_rule_list = [
            FuzzyRule([m_neg_set, lr_neg_set], angle_neg_set),
            FuzzyRule([m_neg_set, lr_mid_set], angle_mid_set),
            FuzzyRule([m_neg_set, lr_pos_set], angle_pos_set),

            FuzzyRule([m_mid_set, lr_neg_set], angle_neg_set),
            FuzzyRule([m_mid_set, lr_mid_set], angle_mid_set),
            FuzzyRule([m_mid_set, lr_pos_set], angle_pos_set),

            FuzzyRule([m_pos_set, lr_neg_set], angle_neg_set),
            FuzzyRule([m_pos_set, lr_mid_set], angle_mid_set),
            FuzzyRule([m_pos_set, lr_pos_set], angle_pos_set)
        ]

        return fuzzy_rule_list


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
        self.start_button = tk.Button(group3, text="Run System", command=self.startBtn_onclick, bg="green")
        self.start_button.grid(row=1, column=0, pady=5)
        # Show Rule button
        self.reset_button = tk.Button(group3, text="Show Fuzzy Sets", command=self.show_fuzzy_sets, bg="light grey")
        self.reset_button.grid(row=2, column=0, pady=5)
        # Save button
        self.save_button = tk.Button(group3, text="Save Car Path", command=self.saveBtn_onclick, bg="grey")
        self.save_button.grid(row=3, column=0, pady=5)
        
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

    def startBtn_onclick(self):
        def state_calucate(center_distance, left_distance, right_distance):
            left_distance = left_distance if left_distance > 0 else 0
            right_distance = right_distance if right_distance > 0 else 0

            center_state = center_distance
            combine_distance = left_distance - right_distance
            return center_distance, combine_distance

        def start_training():
            self.start_button.config(text="Training...", bg="grey", state="disabled")
            self.msg.config(text="Training...", fg="black")

            p = self.playground

            sensor_output = p.reset()
            center_distance, left_distance, right_distance = sensor_output
            center_state, combine_state = state_calucate(center_distance, left_distance, right_distance)
            self.action_list = []
            while not p.done:

                # Fuzzy System
                action = self.fuzzy.infer((center_state, combine_state))
                print(f"pre_state: {center_state}, {combine_state}")
                print(f"action angle: {action}")
                next_sensor_output = p.step(action)
                next_center_distance, next_left_distance, next_right_distance = next_sensor_output
                car_pos = [p.car.getPosition("center").x, p.car.getPosition("center").y]

                next_center_state, next_combine_state = state_calucate(next_center_distance, next_left_distance, next_right_distance)

                center_state, combine_state = next_center_state, next_combine_state
                self.action_list.append({
                    "previous_car_pos": car_pos,
                    "pre_state": [next_center_distance, next_left_distance, next_right_distance],
                    "degree": action
                })
                    
                if p.done:
                    break
            
            if p.isAtDestination:
                print("success")
                self.msg.config(text="Arrived at Destination", fg="black")
            else:
                print("fail")
                self.msg.config(text="Crashed", fg="red")

            self.draw_run()
            print("Running Ended")

            # except Exception as e:
            #     self.msg.config(text=str(e), fg="red")

            self.start_button.config(text="Start Running", bg="green", state="normal")
            self.msg.config(text="Running Ended", fg="black")

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
    
    def show_fuzzy_sets(self) -> None:  # bad
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        
        fuzzy_sets = set()
        for rule in self.fuzzy_rules:
            for fuzzy_set in rule.antecedents + [rule.consequent]:
                fuzzy_sets.add(fuzzy_set)

        for fuzzy_set in fuzzy_sets:
            if fuzzy_set.id.startswith("m_"):
                axs[0].plot(np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET), fuzzy_set.membership_function, label=fuzzy_set.name)
            elif fuzzy_set.id.startswith("lr_"):
                axs[1].plot(np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET), fuzzy_set.membership_function, label=fuzzy_set.name)
            elif fuzzy_set.id.startswith("angle"):
                axs[2].plot(np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET), fuzzy_set.membership_function, label=fuzzy_set.name)

        axs[0].legend()
        axs[0].set_xticks(np.arange(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, 1))
        axs[0].set_title("Center Sensor Distance")

        
        axs[1].legend()
        axs[1].set_xticks(np.arange(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, 1))
        axs[1].set_title("Left - Right Sensor Distance")

        
        axs[2].legend()
        axs[2].set_xticks(np.arange(TURN_LEFT, TURN_RIGHT, 10))
        axs[2].set_title("Wheel Angle")

        plt.tight_layout()
        plt.show()
