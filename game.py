import tkinter as tk
import math
import random

from driver import Driver
from runner import Runner


class Game:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Игра шофёр-убийца — автоматический режим")

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
        self.l = 35.0

        self.driver_turn_gain = 2.2
        self.driver_deadzone = 0.03

        self.driver = Driver()
        self.runner = Runner()

        self.x = 0.0
        self.y = 0.0

        self.paused = False
        self.game_over = False
        self.winner_text = ""

        self.driver_trail = []
        self.runner_trail = []

        self.root.bind("<space>", self.toggle_pause)
        self.root.bind("<r>", self.reset_game)
        self.root.bind("<R>", self.reset_game)

        self.reset_game()

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def sign(value):
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0

    def toggle_pause(self, event=None):
        self.paused = not self.paused

    def reset_game(self, event=None):
        self.driver = Driver()
        self.runner = Runner()

        margin = 80

        self.driver.x = random.uniform(margin, self.canvas_width - margin)
        self.driver.y = random.uniform(margin, self.canvas_height - margin)

        self.driver.theta = random.uniform(0, 2 * math.pi)
        self.driver.phi = 0.0

        min_dist = 100

        while True:
            rx = random.uniform(margin, self.canvas_width - margin)
            ry = random.uniform(margin, self.canvas_height - margin)

            dx = rx - self.driver.x
            dy = ry - self.driver.y
            dist = math.hypot(dx, dy)

            if dist > min_dist:
                break

        self.runner.x = rx
        self.runner.y = ry

        self.runner.psi = random.uniform(0, 2 * math.pi)

        self.x = 0.0
        self.y = 0.0

        self.paused = False
        self.game_over = False
        self.winner_text = ""

        self.driver_trail = []
        self.runner_trail = []

        self.update_reduced_coordinates()

    def keep_inside(self, obj, margin=15):
        if obj.x < margin:
            obj.x = margin
        if obj.x > self.canvas_width - margin:
            obj.x = self.canvas_width - margin
        if obj.y < margin:
            obj.y = margin
        if obj.y > self.canvas_height - margin:
            obj.y = self.canvas_height - margin

    def update_reduced_coordinates(self):
        dx_world = self.runner.x - self.driver.x
        dy_world = self.driver.y - self.runner.y

        sin_t = math.sin(self.driver.theta)
        cos_t = math.cos(self.driver.theta)

        self.x = dx_world * cos_t - dy_world * sin_t
        self.y = dx_world * sin_t + dy_world * cos_t

    def reduced_to_world_runner_angle(self, u_red_x, u_red_y):
        sin_t = math.sin(self.driver.theta)
        cos_t = math.cos(self.driver.theta)

        vx_world = u_red_x * cos_t + u_red_y * sin_t
        vy_world = u_red_x * sin_t - u_red_y * cos_t

        return math.atan2(vy_world, vx_world)

    def compute_runner_autopilot(self):
        r = math.hypot(self.x, self.y)
        if r < 1e-9:
            u_red_x = 0.0
            u_red_y = 1.0
        else:
            u_red_x = self.x / r
            u_red_y = self.y / r

        self.runner.psi = self.reduced_to_world_runner_angle(u_red_x, u_red_y)

    def compute_driver_autopilot(self):
        alpha = math.atan2(self.x, self.y)

        if abs(alpha) < self.driver_deadzone:
            phi = 0.0
        else:
            phi = self.driver_turn_gain * alpha

        self.driver.phi = self.clamp(phi, -1.0, 1.0)

    def compute_autopilot_controls(self):
        self.update_reduced_coordinates()

        self.compute_runner_autopilot()
        self.compute_driver_autopilot()

    def update_absolute_motion(self):
        self.compute_autopilot_controls()

        self.driver.theta += (self.w1 / self.R) * self.driver.phi * self.dt
        self.driver.x += self.w1 * math.sin(self.driver.theta) * self.dt
        self.driver.y -= self.w1 * math.cos(self.driver.theta) * self.dt

        self.runner.x += self.w2 * math.cos(self.runner.psi) * self.dt
        self.runner.y += self.w2 * math.sin(self.runner.psi) * self.dt

        self.keep_inside(self.driver)
        self.keep_inside(self.runner)

        self.update_reduced_coordinates()

    def check_capture(self):
        r = math.hypot(self.x, self.y)
        if r <= self.l:
            self.game_over = True
            self.winner_text = "Шофёр поймал бегущего"

    def update(self):
        if self.paused or self.game_over:
            return

        self.update_absolute_motion()
        self.check_capture()

        self.driver_trail.append((self.driver.x, self.driver.y))
        self.runner_trail.append((self.runner.x, self.runner.y))

        if len(self.driver_trail) > 1000:
            self.driver_trail.pop(0)
        if len(self.runner_trail) > 1000:
            self.runner_trail.pop(0)

    def draw_grid(self):
        step = 50
        for x in range(0, self.canvas_width, step):
            self.canvas.create_line(x, 0, x, self.canvas_height, fill="#eeeeee")
        for y in range(0, self.canvas_height, step):
            self.canvas.create_line(0, y, self.canvas_width, y, fill="#eeeeee")

    def draw_trails(self):
        if len(self.driver_trail) > 1:
            for i in range(1, len(self.driver_trail)):
                x1, y1 = self.driver_trail[i - 1]
                x2, y2 = self.driver_trail[i]
                self.canvas.create_line(x1, y1, x2, y2, fill="#7aa6ff", width=2)

        if len(self.runner_trail) > 1:
            for i in range(1, len(self.runner_trail)):
                x1, y1 = self.runner_trail[i - 1]
                x2, y2 = self.runner_trail[i]
                self.canvas.create_line(x1, y1, x2, y2, fill="#ff8a8a", width=2)

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
        r = math.hypot(self.x, self.y)

        info_lines = [
            "Автоматический режим",
            "Space — пауза",
            "R — reset",
            "",
            f"P = ({self.driver.x:.2f}, {self.driver.y:.2f})",
            f"E = ({self.runner.x:.2f}, {self.runner.y:.2f})",
            "",
            f"x = {self.x:.2f}",
            f"y = {self.y:.2f}",
            f"r = {r:.2f}",
            f"l = {self.l:.2f}",
            "",
            f"phi = {self.driver.phi:.3f}",
            f"psi = {self.runner.psi:.3f}",
            f"theta = {self.driver.theta:.3f}",
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
        box_x = 700
        box_y = 30
        box_w = 270
        box_h = 270

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

        scale = 0.55

        rl = self.l * scale
        self.canvas.create_oval(
            cx - rl, cy - rl,
            cx + rl, cy + rl,
            outline="green", width=2
        )

        self.canvas.create_text(
            cx + rl + 18, cy,
            text="E",
            fill="green",
            font=("Arial", 10, "bold")
        )

        px = cx
        py = cy
        ex = cx + self.x * scale
        ey = cy - self.y * scale

        self.canvas.create_oval(px - 8, py - 8, px + 8, py + 8, fill="blue")
        self.canvas.create_text(px, py - 15, text="P", fill="blue")

        self.canvas.create_oval(ex - 7, ey - 7, ex + 7, ey + 7, fill="red")
        self.canvas.create_text(ex, ey - 15, text="E", fill="red")

        self.canvas.create_line(px, py, ex, ey, fill="#666666", dash=(4, 2))

    def draw_status(self):
        if self.paused:
            self.canvas.create_text(
                self.canvas_width / 2,
                35,
                text="ПАУЗА",
                fill="orange",
                font=("Arial", 20, "bold")
            )

        if self.game_over:
            self.canvas.create_rectangle(
                260, 300, 740, 390,
                fill="#ffffdd", outline="black", width=2
            )
            self.canvas.create_text(
                500, 330,
                text=self.winner_text,
                fill="darkgreen",
                font=("Arial", 22, "bold")
            )
            self.canvas.create_text(
                500, 365,
                text="Нажмите R для перезапуска",
                fill="black",
                font=("Arial", 14)
            )

    def draw(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_trails()
        self.draw_relative_vector()
        self.draw_driver()
        self.draw_runner()
        self.draw_info()
        self.draw_reduced_preview()
        self.draw_status()

    def loop(self):
        self.update()
        self.draw()
        self.root.after(int(self.dt * 1000), self.loop)

    def run(self):
        self.loop()
        self.root.mainloop()


if __name__ == "__main__":
    game = Game()
    game.run()