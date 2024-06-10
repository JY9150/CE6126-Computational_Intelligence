import os

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from MyLib.simple_playground import Playground
from MyLib.Model import LinearModel
from MyLib.ActivactionFunction import ReLu
from MyLib.pso import PSO


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

# PSO Variable
NUMBER_OF_ITERATION = 1000
NUMBER_OF_PARTICLE = 10
RO1 = 0.1
RO2 = 0.7


class App():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PSO")

        self.playground_path = ROOT_PATH + "\\playground\\軌道座標點.txt"
        self.track_path = ROOT_PATH + "\\track\\track4D.txt"

        self.default_model = LinearModel(3, 1, [30], ReLu())
        self.model = self.default_model
        self.playground = Playground(self.playground_path)
        self.animation = None 

        # UI
        self.msg = tk.Label(self.root, text="Particle Swarm Optimization", font=('Arial', 16))
        self.msg.grid(row=FIGURE_SIZE+4, column=0, columnspan=FIGURE_SIZE, rowspan=FIGURE_SIZE)


        self.init_plot_pannel()
        self.init_control_panel()


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
        self.start_button = tk.Button(group3, text="Start PSO", command=self.startBtn_onclick, bg="green")
        self.start_button.grid(row=1, column=0, pady=5)
        # Show Rule button
        self.reset_button = tk.Button(group3, text="Reset Model", command=self.reset_model, bg="light grey")
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
        def finess_func(x: np.ndarray):
            model = self.default_model
            model.setWeights_1d(x)

            p = self.playground
            sensor_output = p.reset()
            center_distance, left_distance, right_distance = sensor_output
            total_reword = 0
            while not p.done:
                wheel_angle = self.model.forward(np.array([center_distance, left_distance, right_distance]))
                next_sensor_output, reword = p.step(wheel_angle)
                center_distance, left_distance, right_distance = next_sensor_output
                total_reword += reword
                if p.done:
                    break
            return total_reword
            

        def start_training():
            self.start_button.config(text="Training...", bg="grey", state="disabled")
            self.msg.config(text="Training...", fg="black")

            model_weights = self.model.getWeights()
            print(model_weights)
            x = np.concatenate([w.flatten() for w in model_weights])
            print(x)

            pso = PSO(len(x), NUMBER_OF_PARTICLE ,NUMBER_OF_ITERATION, finess_func, ro1=RO1, ro2=RO2)
            optimized_x = pso.run()
            # self.model.layer_list[0].weights = optimized_x[:self.model.layer_list[0].weights.size].reshape(self.model.layer_list[0].weights.shape)


            self.model.setWeights_1d(optimized_x)
            p = self.playground

            sensor_output = p.reset()
            center_distance, left_distance, right_distance = sensor_output
            self.action_list = []
            while not p.done:
                action = self.model.forward(np.array([center_distance, left_distance, right_distance]))
                next_sensor_output, reword = p.step(action)


                car_pos = [p.car.getPosition("center").x, p.car.getPosition("center").y]
                self.action_list.append({
                    "previous_car_pos": car_pos,
                    "pre_state": [center_distance, left_distance, right_distance],
                    "degree": action
                })
                center_distance, left_distance, right_distance = next_sensor_output
                    
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

            self.start_button.config(text="Start PSO", bg="green", state="normal")
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
    
    def reset_model(self) -> None:
        self.model = self.default_model
