import tkinter as tk
import math

from driver import Driver
from runner import Runner


class Game:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Игра шофёр-убийца")

        self.canvas_width = 1000
        self.canvas_height = 750

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white"
        )
        self.canvas.pack()

        self.dt = 0.02

        self.w1 = 120.0
        self.w2 = 100.0
        self.R = 120.0

        self.driver = Driver()
        self.runner = Runner()

        self.x = 0.0
        self.y = 0.0

        self.keys = set()
        self.control_mode = "runner"

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.bind("<Tab>", self.switch_control)
        self.root.bind("<r>", self.reset_game)
        self.root.bind("<R>", self.reset_game)

        self.loop()

    def reset_game(self, event=None):
        self.driver = Driver()
        self.runner = Runner()
        self.x = 0.0
        self.y = 0.0
        self.control_mode = "runner"

    def on_key_press(self, event):
        self.keys.add(event.keysym)

    def on_key_release(self, event):
        if event.keysym in self.keys:
            self.keys.remove(event.keysym)

    def switch_control(self, event=None):
        if self.control_mode == "runner":
            self.control_mode = "driver"
        else:
            self.control_mode = "runner"
        return "break"

    def keep_inside(self, obj, margin=15):
        if obj.x < margin:
            obj.x = margin
        if obj.x > self.canvas_width - margin:
            obj.x = self.canvas_width - margin
        if obj.y < margin:
            obj.y = margin
        if obj.y > self.canvas_height - margin:
            obj.y = self.canvas_height - margin

    def update_absolute_motion(self):
        if self.control_mode == "driver":
            self.driver.update(self.keys)
        else:
            self.runner.update(self.keys)

        self.driver.theta += (self.w1 / self.R) * self.driver.phi * self.dt
        self.driver.x += self.w1 * math.sin(self.driver.theta) * self.dt
        self.driver.y -= self.w1 * math.cos(self.driver.theta) * self.dt

        self.runner.x += self.w2 * math.cos(self.runner.psi) * self.dt
        self.runner.y += self.w2 * math.sin(self.runner.psi) * self.dt

        self.keep_inside(self.driver)
        self.keep_inside(self.runner)

    def update_reduced_coordinates(self):
        dx_world = self.runner.x - self.driver.x
        dy_world = self.driver.y - self.runner.y

        sin_t = math.sin(self.driver.theta)
        cos_t = math.cos(self.driver.theta)

        self.x = dx_world * cos_t - dy_world * sin_t
        self.y = dx_world * sin_t + dy_world * cos_t

    def update(self):
        self.update_absolute_motion()
        self.update_reduced_coordinates()

    def draw_grid(self):
        step = 50
        for x in range(0, self.canvas_width, step):
            self.canvas.create_line(x, 0, x, self.canvas_height, fill="#eeeeee")
        for y in range(0, self.canvas_height, step):
            self.canvas.create_line(0, y, self.canvas_width, y, fill="#eeeeee")

    def draw_driver(self):
        px = self.driver.x
        py = self.driver.y
        r = 12

        self.canvas.create_oval(
            px - r, py - r, px + r, py + r,
            fill="blue", outline="black", width=2
        )

        arrow_len = 40
        ax = px + arrow_len * math.sin(self.driver.theta)
        ay = py - arrow_len * math.cos(self.driver.theta)

        self.canvas.create_line(
            px, py, ax, ay,
            fill="navy", width=3, arrow=tk.LAST
        )

        self.canvas.create_text(
            px, py - 20,
            text="P",
            fill="blue",
            font=("Arial", 10, "bold")
        )

    def draw_runner(self):
        ex = self.runner.x
        ey = self.runner.y
        r = 10

        self.canvas.create_oval(
            ex - r, ey - r, ex + r, ey + r,
            fill="red", outline="black", width=2
        )

        arrow_len = 35
        ax = ex + arrow_len * math.cos(self.runner.psi)
        ay = ey + arrow_len * math.sin(self.runner.psi)

        self.canvas.create_line(
            ex, ey, ax, ay,
            fill="darkred", width=2, arrow=tk.LAST
        )

        self.canvas.create_text(
            ex, ey - 20,
            text="E",
            fill="red",
            font=("Arial", 10, "bold")
        )

    def draw_relative_vector(self):
        self.canvas.create_line(
            self.driver.x, self.driver.y,
            self.runner.x, self.runner.y,
            fill="#888888", dash=(5, 3)
        )

    def draw_info(self):
        info_lines = [
            "Управление:",
            "tab — переключить управление",
            "R — reset",
            "Пешеход: вверх вниз вправо влево",
            "Машина: A / D",
            "",
            f"Текущий режим: {self.control_mode}",
            "",
            f"P = ({self.driver.x:.2f}, {self.driver.y:.2f})",
            f"E = ({self.runner.x:.2f}, {self.runner.y:.2f})",
            "",
            f"x = {self.x:.2f}",
            f"y = {self.y:.2f}",
            "",
            f"phi = {self.driver.phi:.2f}",
            f"psi = {self.runner.psi:.2f}",
            f"theta = {self.driver.theta:.2f}",
            "",
            f"w1 = {self.w1}",
            f"w2 = {self.w2}",
            f"R = {self.R}",
        ]

        y0 = 20
        for line in info_lines:
            self.canvas.create_text(
                20, y0,
                anchor="nw",
                text=line,
                fill="black",
                font=("Arial", 12)
            )
            y0 += 22

    def draw_reduced_preview(self):
        box_x = 720
        box_y = 40
        box_w = 240
        box_h = 240

        self.canvas.create_rectangle(
            box_x, box_y,
            box_x + box_w, box_y + box_h,
            outline="black", width=2
        )

        self.canvas.create_text(
            box_x + box_w / 2, box_y - 15,
            text="Редуцированное пространство",
            font=("Arial", 11, "bold")
        )

        cx = box_x + box_w / 2
        cy = box_y + box_h / 2

        self.canvas.create_line(box_x, cy, box_x + box_w, cy, fill="gray")
        self.canvas.create_line(cx, box_y, cx, box_y + box_h, fill="gray")

        scale = 0.5

        px = cx
        py = cy
        ex = cx + self.x * scale
        ey = cy - self.y * scale

        self.canvas.create_oval(px - 8, py - 8, px + 8, py + 8, fill="blue")
        self.canvas.create_text(px, py - 15, text="P", fill="blue")

        self.canvas.create_oval(ex - 7, ey - 7, ex + 7, ey + 7, fill="red")
        self.canvas.create_text(ex, ey - 15, text="E", fill="red")

    def draw(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_relative_vector()
        self.draw_driver()
        self.draw_runner()
        self.draw_info()
        self.draw_reduced_preview()

    def loop(self):
        self.update()
        self.draw()
        self.root.after(int(self.dt * 1000), self.loop)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    game = Game()
    game.run()