import tkinter as tk
import math
import json
import threading
from pathlib import Path

from .driver import Driver
from .runner import Runner
from .arena_policy import (ArenaPolicyConfig, ReducedArenaPolicy, DATA_FILE,
                           load_or_build, reduced_solver_config)
from .pursuit3d import Pursuit3D
from .wall_game import WallPursuitGame, WallValueGrid


class Game:
    ARENA_WIDTH = 800.0
    ARENA_HEIGHT = 600.0

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Игра шофёр-убийца")
        self.root.geometry("1000x750+20+70")

        self.canvas_width = 1000
        self.canvas_height = 750

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white"
        )
        self.canvas.pack()
        self.canvas.configure(takefocus=True)
        # Нашел ошибку ... окно не передавало фокус игровому холсту автоматически,
        # из-за чего клавиши управления могли игнорироваться сразу после запуска.
        self.canvas.focus_set()

        self.dt = 0.02

        self.w1 = 120.0
        self.w2 = 100.0
        # R = 60: l/R ~ 0.58 > sqrt(1-g^2)+arcsin(g)-1 ~ 0.54 (g = w2/w1), т.е. по
        # Айзексу захват гарантирован из любого положения редуцированной плоскости.
        self.R = 60.0
        self.capture_radius = 35.0

        self.driver = Driver()
        self.runner = Runner()
        self._driver_motion = (0.0, 0.0)
        self._runner_motion = (0.0, 0.0)
        self.model_3d = Pursuit3D(w1=self.w1, w2=self.w2, turn_radius=self.R)
        self.wall_game = WallPursuitGame(w1=self.w1, w2=self.w2,
                                         capture_radius=self.capture_radius)
        self.wall_grid = WallValueGrid(w1=self.w1, w2=self.w2,
                                       capture_radius=self.capture_radius)
        self.mode_3d = False
        self.wall_mode = False
        self.arena_mode = True
        self.captured = False
        self.status_message = ""
        self.arena_hji = None
        self._hji_lock = threading.Lock()
        self._hji_progress = ("ожидание", 0, 0, 0.0)
        self._hji_state = "загружается"
        self._hji_thread = None

        self.x = 0.0
        self.y = 0.0

        self.keys = set()
        self.control_mode = "runner"
        # Храним ограниченную историю, чтобы можно было видеть пройденный путь.
        self.trail_limit = 1800
        self.trail_sample_every = 4
        self._trail_tick = 0
        self.driver_trail = [(self.driver.x, self.driver.y)]
        self.runner_trail = [(self.runner.x, self.runner.y)]
        self.driver_trail_3d = [tuple(self.model_3d.pursuer_position)]
        self.runner_trail_3d = [tuple(self.model_3d.evader_position)]
        self.wall_pursuer_trail = [self.wall_game.pursuer_world_position]
        self.wall_evader_trail = [self.wall_game.evader_world_position]

        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.root.bind_all("<KeyRelease>", self.on_key_release)
        self.root.bind("<Tab>", self.switch_control)
        self.root.protocol("WM_DELETE_WINDOW", self.close_game)
        self.create_dashboard()
        self.ensure_arena_hji()

        self.loop()

    def _arena_hji_config(self):
        return reduced_solver_config(w1=self.w1, w2=self.w2, turn_radius=self.R,
                                     capture_radius=self.capture_radius)

    def _make_arena_policy(self, solver):
        return ReducedArenaPolicy(solver, ArenaPolicyConfig(width=self.ARENA_WIDTH,
                                                            height=self.ARENA_HEIGHT))

    def ensure_arena_hji(self):
        """Загрузить редуцированную сетку V(x, y) или посчитать её в фоне."""
        expected = self._arena_hji_config()
        solver = load_or_build(DATA_FILE, expected, build=False)
        if solver is not None:
            self.arena_hji = self._make_arena_policy(solver)
            self._hji_state = "редуцированная сетка V(x, y) загружена из файла"
            return
        self._hji_state = "редуцированная сетка рассчитывается в фоне; ручное управление доступно"
        self._hji_thread = threading.Thread(
            target=self._solve_arena_hji, args=(DATA_FILE, expected),
            name="reduced-hji", daemon=True
        )
        self._hji_thread.start()

    def _solve_arena_hji(self, solution_file, config, seed=None):
        try:
            with self._hji_lock:
                self._hji_progress = ("итерации HJI", 0, config.max_iterations, 0.0)
            solver = load_or_build(solution_file, config, build=True)
            self.arena_hji = self._make_arena_policy(solver)
            with self._hji_lock:
                self._hji_progress = ("итерации HJI", solver.iterations,
                                      config.max_iterations, solver.residual)
            self._hji_state = ("готова" if solver.converged
                               else "предварительная; лимит итераций достигнут")
        except ImportError:
            self._hji_state = "для расчёта сетки нужен NumPy (pip install numpy)"
        except Exception as exc:  # Keep the manual game usable if the grid fails.
            self._hji_state = f"ошибка расчёта: {exc}"

    def create_dashboard(self):
        """Создает отдельное окно для телеметрии и органов управления."""
        self.dashboard = tk.Toplevel(self.root)
        self.dashboard.title("Погоня — состояние и управление")
        self.dashboard.geometry("480x900+1030+60")
        self.dashboard.minsize(400, 500)
        self.dashboard.protocol("WM_DELETE_WINDOW", self.close_game)

        outer = tk.Frame(self.dashboard, padx=16, pady=14)
        outer.pack(fill="both", expand=True)
        tk.Label(outer, text="ПАНЕЛЬ НАБЛЮДЕНИЯ", font=("Arial", 16, "bold")).pack(anchor="w")
        tk.Label(outer, text="Данные обновляются вместе с симуляцией",
                 font=("Arial", 10), fg="#555555").pack(anchor="w", pady=(2, 12))

        controls = tk.LabelFrame(outer, text="Управление", padx=10, pady=8)
        controls.pack(fill="x", pady=(0, 12))
        buttons = [
            ("Передать управление (Tab)", self.switch_control),
            ("Авто HJI (H)", self.toggle_auto),
            ("Арена 2D", self.enable_2d),
            ("Игра у стены (1)", self.enable_wall_game),
            ("3D", self.enable_3d),
            ("Сбросить (R)", self.reset_game),
        ]
        for index, (label, action) in enumerate(buttons):
            tk.Button(controls, text=label, command=action, padx=8, pady=5).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=4, pady=4
            )
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        tk.Label(outer, text="Состояние в реальном времени", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(0, 6)
        )
        self.dashboard_text = tk.StringVar()
        tk.Label(outer, textvariable=self.dashboard_text, justify="left", anchor="nw",
                 wraplength=420, font=("Consolas", 10), bg="#f4f6f8", padx=10, pady=8,
                 relief="groove").pack(fill="x", expand=False)
        tk.Label(outer, text="Редуцированное состояние (x, y)",
                 font=("Arial", 12, "bold")).pack(anchor="w", pady=(12, 5))
        self.reduced_canvas = tk.Canvas(outer, width=430, height=300,
                                        bg="white", highlightthickness=1,
                                        highlightbackground="#999999")
        self.reduced_canvas.pack(fill="both", expand=True)
        tk.Label(outer,
                 text="Клавиши: Tab — P/E | P: A/D | E: стрелки | 1: стена | 2: арена | R: сброс | H: HJI",
                 justify="left", wraplength=420, fg="#444444").pack(anchor="w", pady=(10, 0))

    def close_game(self):
        self.root.destroy()

    def reset_game(self, event=None):
        self.driver = Driver()
        self.runner = Runner()
        self._driver_motion = (0.0, 0.0)
        self._runner_motion = (0.0, 0.0)
        self.model_3d = Pursuit3D(w1=self.w1, w2=self.w2, turn_radius=self.R)
        self.wall_game = WallPursuitGame(w1=self.w1, w2=self.w2,
                                         capture_radius=self.capture_radius)
        self.x = 0.0
        self.y = 0.0
        self.control_mode = "runner"
        self.mode_3d = False
        self.wall_mode = False
        self.arena_mode = True
        self.captured = False
        self.status_message = ""
        self._trail_tick = 0
        self.driver_trail = [(self.driver.x, self.driver.y)]
        self.runner_trail = [(self.runner.x, self.runner.y)]
        self.driver_trail_3d = [tuple(self.model_3d.pursuer_position)]
        self.runner_trail_3d = [tuple(self.model_3d.evader_position)]
        self.wall_pursuer_trail = [self.wall_game.pursuer_world_position]
        self.wall_evader_trail = [self.wall_game.evader_world_position]

    def toggle_auto(self, event=None):
        if self.mode_3d or self.wall_mode:
            return "break"
        if self.control_mode == "auto":
            self.control_mode = "runner"
        elif self.arena_hji is not None and self.arena_hji.converged:
            if self.arena_hji.value_at(self.driver.x, self.driver.y,
                                       self.runner.x, self.runner.y,
                                       self.driver.theta) < self.arena_hji.cap:
                self.status_message = ""
                self.control_mode = "auto"
            else:
                self.status_message = "Текущее состояние вне выигрышной области сетки HJI"
        else:
            self.status_message = "HJI ещё считается/не сошлась: доступно ручное управление"
        return "break"

    def enable_3d(self, event=None):
        self.wall_mode = False
        self.mode_3d = True
        self.arena_mode = False
        self.control_mode = "runner"
        self.captured = False
        return "break"

    def enable_2d(self, event=None):
        if self.mode_3d:
            self.driver.x, self.driver.y = self.model_3d.pursuer_position[:2]
            self.runner.x, self.runner.y = self.model_3d.evader_position[:2]
            self.driver.theta = self.model_3d.yaw
            self.mode_3d = False
        self.wall_mode = False
        self.arena_mode = True
        return "break"

    def enable_wall_game(self, event=None):
        """Select the distinct half-plane wall game and its fixed-coordinate grid."""
        self.mode_3d = False
        self.wall_mode = True
        self.arena_mode = False
        self.control_mode = "wall-auto"
        self.captured = False
        self.status_message = "Одна непроходимая стена; отдельная 2D модель Айзекса"
        self.wall_game = WallPursuitGame(w1=self.w1, w2=self.w2,
                                         capture_radius=self.capture_radius)
        self.wall_pursuer_trail = [self.wall_game.pursuer_world_position]
        self.wall_evader_trail = [self.wall_game.evader_world_position]
        return "break"

    def enable_chauffeur_game(self, event=None):
        return self.enable_2d(event)

    def on_key_press(self, event):
        self.keys.add(event.keysym)
        # Нашел ошибку ... буквенно-цифровые команды дублировались отдельными
        # bindings и терялись при смене регистра; диспетчеризуем их из KeyPress.
        key = event.keysym.lower()
        if key == "r":
            self.reset_game()
            return "break"
        if key == "tab":
            return self.switch_control()
        if key == "h":
            return self.toggle_auto()
        if key == "3":
            return self.enable_3d()
        if key == "2":
            return self.enable_2d()
        if key == "1":
            return self.enable_wall_game()

    def on_key_release(self, event):
        if event.keysym in self.keys:
            self.keys.remove(event.keysym)

    def switch_control(self, event=None):
        if self.control_mode == "runner":
            self.control_mode = "driver"
        else:
            self.control_mode = "runner"
        return "break"

    def _record_trails(self):
        self._trail_tick += 1
        if self._trail_tick % self.trail_sample_every:
            return
        if self.wall_mode:
            self.wall_pursuer_trail.append(self.wall_game.pursuer_world_position)
            self.wall_evader_trail.append(self.wall_game.evader_world_position)
            self.wall_pursuer_trail = self.wall_pursuer_trail[-self.trail_limit:]
            self.wall_evader_trail = self.wall_evader_trail[-self.trail_limit:]
        elif self.mode_3d:
            self.driver_trail_3d.append(tuple(self.model_3d.pursuer_position))
            self.runner_trail_3d.append(tuple(self.model_3d.evader_position))
            self.driver_trail_3d = self.driver_trail_3d[-self.trail_limit:]
            self.runner_trail_3d = self.runner_trail_3d[-self.trail_limit:]
        else:
            self.driver_trail.append((self.driver.x, self.driver.y))
            self.runner_trail.append((self.runner.x, self.runner.y))
            self.driver_trail = self.driver_trail[-self.trail_limit:]
            self.runner_trail = self.runner_trail[-self.trail_limit:]

    def update_absolute_motion(self):
        if self.captured:
            if not self.wall_mode and not self.mode_3d:
                self._driver_motion = self._runner_motion = (0.0, 0.0)
            return
        if self.wall_mode:
            self.wall_game.step(self.dt, self.wall_grid)
            self.captured = self.wall_game.captured
            self._record_trails()
            return
        if self.mode_3d:
            if self.control_mode == "driver":
                yaw = ("d" in self.keys or "D" in self.keys) - ("a" in self.keys or "A" in self.keys)
                pitch = ("w" in self.keys or "W" in self.keys) - ("s" in self.keys or "S" in self.keys)
                azimuth = elevation = 0.0
            else:
                yaw = pitch = 0.0
                azimuth = ("Right" in self.keys) - ("Left" in self.keys)
                elevation = ("Up" in self.keys) - ("Down" in self.keys)
            self.model_3d.step(yaw, pitch, azimuth, elevation, self.dt)
            self.captured = self.model_3d.captured
            self._record_trails()
            return

        if self.control_mode == "auto":
            if (self.arena_hji is None or not self.arena_hji.converged
                    or self.arena_hji.value_at(
                        self.driver.x, self.driver.y, self.runner.x, self.runner.y,
                        self.driver.theta
                    ) >= self.arena_hji.cap):
                self.control_mode = "runner"
                self.status_message = "HJI остановлена: сетка не сошлась или состояние вне её области"
                self.runner.update(self.keys)
            else:
                # Нашел ошибку ... удержание решения весь HJI-шаг 0.5 с
                # нарушало лимит горизонта и увеличивало время захвата; в игре
                # политика пересчитывается на каждом физическом шаге.
                self.driver.phi, self.runner.psi = self.arena_hji.optimal_controls(
                    self.driver.x, self.driver.y, self.runner.x, self.runner.y,
                    self.driver.theta
                )
        elif self.control_mode == "driver":
            self.driver.update(self.keys)
        else:
            self.runner.update(self.keys)

        old_driver_position = (self.driver.x, self.driver.y)
        old_runner_position = (self.runner.x, self.runner.y)
        self.driver.theta += (self.w1 / self.R) * self.driver.phi * self.dt
        self.driver.x += self.w1 * math.sin(self.driver.theta) * self.dt
        self.driver.y -= self.w1 * math.cos(self.driver.theta) * self.dt

        self.runner.x += self.w2 * math.sin(self.runner.psi) * self.dt
        self.runner.y -= self.w2 * math.cos(self.runner.psi) * self.dt

        # Нашел ошибку ... прежний режим был безграничным; теперь стеновые
        # ограничения применяются после каждого шага ко всем четырём координатам.
        self.constrain_players_to_arena()
        # Сохраняем именно фактическое смещение после стенового ограничения:
        # у границы оно может отличаться от заданного курса или быть нулевым.
        self._driver_motion = (self.driver.x - old_driver_position[0],
                               self.driver.y - old_driver_position[1])
        self._runner_motion = (self.runner.x - old_runner_position[0],
                               self.runner.y - old_runner_position[1])
        self.captured = math.hypot(self.runner.x - self.driver.x,
                                   self.driver.y - self.runner.y) <= self.capture_radius
        self._record_trails()

    def constrain_players_to_arena(self):
        if not self.arena_mode:
            return
        for player in (self.driver, self.runner):
            player.x = min(self.ARENA_WIDTH, max(0.0, player.x))
            player.y = min(self.ARENA_HEIGHT, max(0.0, player.y))

    def update_reduced_coordinates(self):
        dx_world = self.runner.x - self.driver.x
        dy_world = self.driver.y - self.runner.y

        sin_t = math.sin(self.driver.theta)
        cos_t = math.cos(self.driver.theta)

        self.x = dx_world * cos_t - dy_world * sin_t
        self.y = dx_world * sin_t + dy_world * cos_t

    def update(self):
        self.update_absolute_motion()
        if not self.mode_3d and not self.wall_mode:
            self.update_reduced_coordinates()

    def draw_grid(self):
        step = 100
        scale = self.camera_scale_2d()
        spacing = step * scale
        if spacing < 24:
            step = 500
            spacing = step * scale
        if spacing > 130:
            step = 50
            spacing = step * scale
        if self.arena_mode:
            # Нашел ошибку ... сетка привязывалась к середине между игроками,
            # поэтому визуально «ехала» при неподвижных стенах. В арене её
            # начало теперь закреплено за мировым центром прямоугольника.
            center_x = self.ARENA_WIDTH / 2
            center_y = self.ARENA_HEIGHT / 2
        else:
            center_x = (self.driver.x + self.runner.x) / 2
            center_y = (self.driver.y + self.runner.y) / 2
        ox, oy = self.world_to_screen(center_x, center_y)
        for x in range(int(ox % spacing), self.canvas_width, max(1, int(spacing))):
            self.canvas.create_line(x, 0, x, self.canvas_height, fill="#eeeeee")
        for y in range(int(oy % spacing), self.canvas_height, max(1, int(spacing))):
            self.canvas.create_line(0, y, self.canvas_width, y, fill="#eeeeee")

    def camera_scale_2d(self):
        if getattr(self, "arena_mode", False):
            pad = 55
            return min((self.canvas_width - 2 * pad) / self.ARENA_WIDTH,
                       (self.canvas_height - 2 * pad) / self.ARENA_HEIGHT)
        dx = abs(self.runner.x - self.driver.x)
        dy = abs(self.runner.y - self.driver.y)
        pad = 70
        return min((self.canvas_width - 2 * pad) / max(dx + 100, 100),
                   (self.canvas_height - 2 * pad) / max(dy + 100, 100), 1.0)

    def world_to_screen(self, x, y):
        # Физика игры использует экранную систему координат: ось y направлена
        # вниз (курс theta = 0 и клавиша «вверх» уменьшают y). Поэтому y при
        # выводе не переворачивается; прежний переворот зеркалил картинку:
        # «вверх» на клавиатуре уводило игрока вниз, а стена y = 0 рисовалась снизу.
        scale = self.camera_scale_2d()
        if getattr(self, "arena_mode", False):
            return (self.canvas_width / 2 + (x - self.ARENA_WIDTH / 2) * scale,
                    self.canvas_height / 2 + (y - self.ARENA_HEIGHT / 2) * scale)
        center_x = (self.driver.x + self.runner.x) / 2
        center_y = (self.driver.y + self.runner.y) / 2
        return (self.canvas_width / 2 + (x - center_x) * scale,
                self.canvas_height / 2 + (y - center_y) * scale)

    def draw_trail(self, points, color):
        if len(points) < 2:
            return
        coords = []
        for x, y in points:
            coords.extend(self.world_to_screen(x, y))
        self.canvas.create_line(*coords, fill=color, width=2, smooth=True,
                                capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def draw_driver(self):
        px, py = self.world_to_screen(self.driver.x, self.driver.y)
        r = 12

        self.canvas.create_oval(
            px - r, py - r, px + r, py + r,
            fill="blue", outline="black", width=2
        )

        motion_x, motion_y = getattr(self, "_driver_motion", (0.0, 0.0))
        motion_len = math.hypot(motion_x, motion_y)
        if motion_len > 1e-9:
            arrow_len = 40
            ax = px + arrow_len * motion_x / motion_len
            ay = py + arrow_len * motion_y / motion_len
            self.canvas.create_line(px, py, ax, ay, fill="navy", width=3, arrow=tk.LAST)
        else:
            # В начале игры курса ещё нет; у стены неподвижность не маскируем
            # стрелкой, направленной сквозь непроходимую границу.
            self.canvas.create_oval(px - 2, py - 2, px + 2, py + 2,
                                    fill="navy", outline="navy")

        self.canvas.create_text(
            px, py - 20,
            text="P",
            fill="blue",
            font=("Arial", 10, "bold")
        )

    def draw_runner(self):
        ex, ey = self.world_to_screen(self.runner.x, self.runner.y)
        r = 10

        self.canvas.create_oval(
            ex - r, ey - r, ex + r, ey + r,
            fill="red", outline="black", width=2
        )

        motion_x, motion_y = getattr(self, "_runner_motion", (0.0, 0.0))
        motion_len = math.hypot(motion_x, motion_y)
        if motion_len > 1e-9:
            arrow_len = 35
            ax = ex + arrow_len * motion_x / motion_len
            ay = ey + arrow_len * motion_y / motion_len
            self.canvas.create_line(ex, ey, ax, ay, fill="darkred", width=2, arrow=tk.LAST)

        self.canvas.create_text(
            ex, ey - 20,
            text="E",
            fill="red",
            font=("Arial", 10, "bold")
        )

    def draw_relative_vector(self):
        self.canvas.create_line(
            *self.world_to_screen(self.driver.x, self.driver.y),
            *self.world_to_screen(self.runner.x, self.runner.y),
            fill="#888888", dash=(5, 3)
        )

    def draw_info(self):
        mode_label = {"runner": "ручное управление E", "driver": "ручное управление P",
                      "auto": "автоматическое HJI (min-max)",
                      "wall-auto": "аналитическая HJI стеновой игры"}.get(
                          self.control_mode, self.control_mode
                      )
        if self.mode_3d:
            p = self.model_3d.pursuer_position
            e = self.model_3d.evader_position
            rx, ry, rz = self.model_3d.relative_position
            distance = self.model_3d.separation
            lines = [
                "МОДЕЛЬ: трехмерная",
                f"Управление: {mode_label}     Захват: {'ДА' if self.captured else 'нет'}",
                "",
                f"P = ({p[0]:9.2f}, {p[1]:9.2f}, {p[2]:9.2f})",
                f"E = ({e[0]:9.2f}, {e[1]:9.2f}, {e[2]:9.2f})",
                f"r = вперед {rx:.2f}, вправо {ry:.2f}, вверх {rz:.2f}",
                f"Расстояние: {distance:.2f}     Время: {self.model_3d.elapsed:.2f} с",
                "",
                f"Пройдено точек траектории: P {len(self.driver_trail_3d)} | E {len(self.runner_trail_3d)}",
                "Автокамера удерживает P и E в кадре.",
                "P: A/D — рыскание, W/S — тангаж",
                "E: стрелки — азимут и высота",
                "Нажмите 2 для возврата в плоскую модель.",
            ]
        elif self.wall_mode:
            game = self.wall_game
            saddle_x = self.wall_grid.saddle_abscissa(game.x, game.y)
            lines = [
                "МОДЕЛЬ: отдельная игра у одной непроходимой стены",
                f"Управление: {mode_label}     Захват: {'ДА' if self.captured else 'нет'}",
                "",
                f"Состояние редуцированной системы: x={game.x:.2f}, y={game.y:.2f}",
                f"P = ({game.pursuer_world_position[0]:.2f}, {game.pursuer_world_position[1]:.2f})",
                f"E = ({game.evader_world_position[0]:.2f}, {game.evader_world_position[1]:.2f})",
                f"Точное время до захвата: {game.value:.3f} с",
                f"Сетка: фиксированный старт {self.wall_grid.start}, шаг {self.wall_grid.spacing:g}",
                f"V_grid(start)={self.wall_grid.value_at(*self.wall_grid.start):.3f} с; ",
                f"ошибка интерполяции={self.wall_grid.refinement_error(*self.wall_grid.start):.2g}",
                f"Поверхность рассеяния y=0; соответствующая точка: ({saddle_x:.2f}, 0)",
                "Седловая поверхность не является целью захвата или устойчивым аттрактором.",
                f"Невязка точной HJI: {game.hamiltonian_residual():.2e}",
                f"Время симуляции: {game.elapsed:.2f} с",
                "Нажмите 2 — вернуться к арене с четырьмя стенами.",
            ]
        else:
            distance = math.hypot(self.runner.x - self.driver.x,
                                  self.runner.y - self.driver.y)
            lines = [
                "МОДЕЛЬ: замкнутая прямоугольная арена",
                f"Управление: {mode_label}     Захват: {'ДА' if self.captured else 'нет'}",
                "",
                f"P = ({self.driver.x:9.2f}, {self.driver.y:9.2f})",
                f"E = ({self.runner.x:9.2f}, {self.runner.y:9.2f})",
                f"Относительное состояние: x={self.x:.2f}, y={self.y:.2f}",
                f"Расстояние: {distance:.2f}     Радиус захвата: {self.capture_radius:.2f}",
                "",
                f"phi={self.driver.phi:.3f}  psi={self.runner.psi:.3f}  theta={self.driver.theta:.3f}",
                f"Параметры: w1={self.w1:g}, w2={self.w2:g}, R={self.R:g}",
                f"Пройдено точек траектории: P {len(self.driver_trail)} | E {len(self.runner_trail)}",
                f"Границы: x=[0,{self.ARENA_WIDTH:g}], y=[0,{self.ARENA_HEIGHT:g}]; выйти нельзя.",
                f"Сетка HJI: {self._hji_state}.",
                f"Расчёт сетки: {self._hji_progress[0]} "
                f"{self._hji_progress[1]}/{self._hji_progress[2]} "
                f"(невязка {self._hji_progress[3]:.3g}).",
            ]
            if self.arena_hji is None:
                lines.append("HJI: пока недоступна — продолжается расчёт редуцированной сетки")
            else:
                start_value = self.arena_hji.value_at(300.0, 500.0, 600.0, 250.0, 0.0)
                value = self.arena_hji.value_at(self.driver.x, self.driver.y,
                                                self.runner.x, self.runner.y,
                                                self.driver.theta)
                lines.append(f"HJI фиксированного старта (P=300,500; E=600,250): "
                             f"Vсетка≈{start_value:.2f} с")
                lines.append(f"HJI: V(x, y)≈{value:.2f} с (редуцированная сетка, без учёта стен)")
        if self.status_message:
            lines.extend(("", f"Сообщение: {self.status_message}"))
        self.dashboard_text.set("\n".join(lines))

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
        if self.wall_mode:
            self.draw_wall_scene()
            return
        if self.mode_3d:
            self.draw_3d()
            return
        self.draw_grid()
        if self.arena_mode:
            self.draw_arena_bounds()
        self.draw_trail(self.driver_trail, "#8aa8e8")
        self.draw_trail(self.runner_trail, "#e9a0a0")
        self.draw_relative_vector()
        self.draw_driver()
        self.draw_runner()

    def draw_arena_bounds(self):
        top_left = self.world_to_screen(0.0, self.ARENA_HEIGHT)
        bottom_right = self.world_to_screen(self.ARENA_WIDTH, 0.0)
        self.canvas.create_rectangle(*top_left, *bottom_right, outline="#242a33", width=5)
        for label, pos, anchor in (("ВЕРХ", (self.ARENA_WIDTH / 2, 0), "n"),
                                   ("НИЗ", (self.ARENA_WIDTH / 2, self.ARENA_HEIGHT), "s"),
                                   ("ЛЕВО", (0, self.ARENA_HEIGHT / 2), "w"),
                                   ("ПРАВО", (self.ARENA_WIDTH, self.ARENA_HEIGHT / 2), "e")):
            sx, sy = self.world_to_screen(*pos)
            self.canvas.create_text(sx, sy, text=label, anchor=anchor,
                                    fill="#333333", font=("Arial", 9, "bold"))

    def draw_wall_scene(self):
        game = self.wall_game
        p = game.pursuer_world_position
        e = game.evader_world_position
        radius = game.capture_radius
        xmin = min(p[0], -radius)
        xmax = radius
        ymin = min(p[1], e[1]) - radius
        ymax = max(p[1], e[1]) + radius
        pad = 70
        scale = min((self.canvas_width - 2 * pad) / max(xmax - xmin, 1.0),
                    (self.canvas_height - 2 * pad) / max(ymax - ymin, 1.0), 1.0)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        def project(point):
            return (self.canvas_width / 2 + (point[0] - center_x) * scale,
                    self.canvas_height / 2 - (point[1] - center_y) * scale)

        for gx in range(math.ceil(xmin / 100) * 100, math.floor(xmax / 100) * 100 + 1, 100):
            sx, _ = project((gx, center_y))
            self.canvas.create_line(sx, 0, sx, self.canvas_height, fill="#eeeeee")
        for gy in range(math.ceil(ymin / 100) * 100, math.floor(ymax / 100) * 100 + 1, 100):
            _, sy = project((center_x, gy))
            self.canvas.create_line(0, sy, self.canvas_width, sy, fill="#eeeeee")

        wall_top = project((0.0, ymax))
        wall_bottom = project((0.0, ymin))
        self.canvas.create_line(*wall_top, *wall_bottom, fill="#333333", width=5)
        self.canvas.create_text(wall_top[0] - 8, 18, text="СТЕНА W", anchor="ne",
                                fill="#333333", font=("Arial", 12, "bold"))

        for trail, color in ((self.wall_pursuer_trail, "#8aa8e8"),
                             (self.wall_evader_trail, "#e9a0a0")):
            if len(trail) > 1:
                coords = [v for point in trail for v in project(point)]
                self.canvas.create_line(*coords, fill=color, width=2, smooth=True)

        arc = [project((e[0] + radius * math.cos(math.pi * (k / 60 - 0.5)),
                        e[1] + radius * math.sin(math.pi * (k / 60 - 0.5))))
               for k in range(61)]
        self.canvas.create_line(*(v for point in arc for v in point),
                                fill="#55aa55", dash=(4, 3), width=2)

        ps = project(p)
        es = project(e)
        self.canvas.create_line(*ps, *es, fill="#888888", dash=(5, 3), width=2)
        self.canvas.create_oval(ps[0] - 12, ps[1] - 12, ps[0] + 12, ps[1] + 12,
                                fill="blue", outline="black", width=2)
        self.canvas.create_text(ps[0], ps[1] - 22, text="P", fill="blue",
                                font=("Arial", 12, "bold"))
        self.canvas.create_oval(es[0] - 10, es[1] - 10, es[0] + 10, es[1] + 10,
                                fill="red", outline="black", width=2)
        self.canvas.create_text(es[0], es[1] - 20, text="E", fill="red",
                                font=("Arial", 12, "bold"))
        heading, evader_direction = game.optimal_controls(self.wall_grid)
        self.canvas.create_line(ps[0], ps[1], ps[0] - 42 * math.cos(heading),
                                ps[1] + 42 * math.sin(heading), fill="navy", width=3,
                                arrow=tk.LAST)
        self.canvas.create_line(es[0], es[1], es[0], es[1] + 32 * evader_direction,
                                fill="darkred", width=2, arrow=tk.LAST)

    def draw_reduced_state(self):
        canvas = self.reduced_canvas
        canvas.delete("all")
        # Use the configured logical canvas size; update_idletasks() here can
        # expose an empty frame between delete() and the drawing operations.
        width = int(canvas.cget("width"))
        height = int(canvas.cget("height"))
        left, right, top, bottom = 42, width - 18, 18, height - 34
        if self.wall_mode:
            state_x, state_y = self.wall_game.x, self.wall_game.y
            extent = max(100.0, abs(state_x) * 1.2, abs(state_y) * 1.3,
                         self.capture_radius * 2.5)
            xmin, xmax, ymin, ymax = 0.0, extent, -extent, extent
        else:
            state_x, state_y = self.x, self.y
            extent = max(100.0, abs(state_x) * 1.2, abs(state_y) * 1.2,
                         self.capture_radius * 2.5)
            xmin, xmax, ymin, ymax = -extent, extent, -extent, extent
        scale = min((right - left) / max(xmax - xmin, 1.0),
                    (bottom - top) / max(ymax - ymin, 1.0))

        def point(x, y):
            return left + (x - xmin) * scale, bottom - (y - ymin) * scale

        canvas.create_rectangle(left, top, right, bottom, outline="#bbbbbb")
        for k in range(math.ceil(xmin / 100) * 100, math.floor(xmax / 100) * 100 + 1, 100):
            sx, _ = point(k, 0)
            canvas.create_line(sx, top, sx, bottom, fill="#eeeeee")
        for k in range(math.ceil(ymin / 100) * 100, math.floor(ymax / 100) * 100 + 1, 100):
            _, sy = point(0, k)
            canvas.create_line(left, sy, right, sy, fill="#eeeeee")
        origin = point(0, 0)
        canvas.create_line(left, origin[1], right, origin[1], fill="#666666")
        if self.wall_mode:
            canvas.create_line(origin[0], top, origin[0], bottom, fill="#333333", width=3)
            canvas.create_text(origin[0] - 5, top + 2, text="W", anchor="ne", fill="#333333")
            angles = [math.pi * (k / 60 - 0.5) for k in range(61)]
            arc = [point(self.capture_radius * math.cos(a),
                         self.capture_radius * math.sin(a)) for a in angles]
            canvas.create_line(*(v for item in arc for v in item), fill="#55aa55", width=2)
            arc_top = point(0, self.capture_radius)
            arc_bottom = point(0, -self.capture_radius)
            canvas.create_line(*arc_top, *arc_bottom, fill="#55aa55", width=2)
            dispersal_start = point(self.capture_radius, 0)
            canvas.create_line(*dispersal_start, right, dispersal_start[1],
                               fill="#d28b00", dash=(5, 3), width=2)
            canvas.create_text(right - 3, dispersal_start[1] - 10,
                               text="рассеивающая поверхность", anchor="e",
                               fill="#a26b00", font=("Arial", 8))
            saddle_x = self.wall_grid.saddle_abscissa(state_x, state_y)
            saddle_point = point(saddle_x, 0.0)
            canvas.create_oval(saddle_point[0] - 5, saddle_point[1] - 5,
                               saddle_point[0] + 5, saddle_point[1] + 5,
                               fill="#d28b00", outline="black")
            canvas.create_text(saddle_point[0] + 4, saddle_point[1] + 9,
                               text="характеристическая точка", anchor="nw",
                               fill="#8a5c00", font=("Arial", 7))
            canvas.create_text(left + 3, top + 2, text="стеновая полуплоскость x≥0",
                               anchor="nw", fill="#444444", font=("Arial", 9))
        else:
            rpx = self.capture_radius * scale
            canvas.create_oval(origin[0] - rpx, origin[1] - rpx,
                               origin[0] + rpx, origin[1] + rpx,
                               outline="#55aa55", width=2)
            canvas.create_text(left + 3, top + 2,
                               text="проекция; HJI также учитывает P, E и курс P",
                               anchor="nw", fill="#444444", font=("Arial", 9))
        state = point(state_x, state_y)
        canvas.create_line(origin[0], origin[1], *state, fill="#888888", dash=(4, 3))
        if not self.wall_mode:
            # Нашел ошибку ... phi и psi раньше показывались у объектов в
            # абсолютной сцене. Здесь отображаем их вклады в скорость
            # редуцированного состояния (xdot, ydot), а не абсолютные курсы.
            omega = (self.w1 / self.R) * self.driver.phi
            phi_vector = (-omega * state_y, omega * state_x)
            psi_relative = self.runner.psi - self.driver.theta
            psi_vector = (self.w2 * math.sin(psi_relative),
                          self.w2 * math.cos(psi_relative))
            for label, (vx, vy), color in (
                    ("φ", phi_vector, "#d97706"),
                    ("ψ", psi_vector, "#7c3aed")):
                magnitude = math.hypot(vx, vy)
                if magnitude > 1e-9:
                    # Длины нормированы, чтобы направление оставалось читаемым
                    # при автоматическом изменении масштаба графика.
                    end_x = state[0] + 34 * vx / magnitude
                    end_y = state[1] - 34 * vy / magnitude
                    canvas.create_line(*state, end_x, end_y, fill=color, width=2,
                                       arrow=tk.LAST)
                    canvas.create_text(end_x + 4, end_y - 3, text=label,
                                       fill=color, font=("Arial", 10, "bold"))
                else:
                    canvas.create_text(state[0] + 8, state[1] + 10,
                                       text=f"{label}=0", fill=color,
                                       font=("Arial", 9, "bold"))
            canvas.create_text(left + 3, bottom - 2,
                               text="Векторы — вклады в (ẋ, ẏ); длина нормирована",
                               anchor="sw", fill="#555555", font=("Arial", 8))
        canvas.create_oval(state[0] - 5, state[1] - 5, state[0] + 5, state[1] + 5,
                           fill="red", outline="black")
        canvas.create_text(state[0] + 7, state[1] - 8,
                           text=f"({state_x:.0f}, {state_y:.0f})", anchor="sw",
                           fill="red", font=("Arial", 9, "bold"))
        canvas.create_text(width / 2, height - 8, text="x", fill="#333333")
        canvas.create_text(12, top, text="y", fill="#333333")

    def draw_3d(self):
        cx, cy = self.canvas_width / 2, self.canvas_height / 2
        p3 = self.model_3d.pursuer_position
        e3 = self.model_3d.evader_position
        center = tuple((p3[i] + e3[i]) / 2 for i in range(3))
        dx, dy, dz = (e3[i] - p3[i] for i in range(3))
        projected_width = abs(dx - dy) * 0.72
        projected_height = abs((dx + dy) * 0.34 - dz)
        scale = min((self.canvas_width - 140) / max(projected_width + 100, 100),
                    (self.canvas_height - 140) / max(projected_height + 100, 100), 0.62)

        def project(p):
            x, y, z = (p[i] - center[i] for i in range(3))
            return cx + scale * (x - y) * 0.72, cy + scale * ((x + y) * 0.34 - z)

        origin = project(center)
        for vector, color, label in (((300, 0, 0), "#cc5555", "X"),
                                     ((0, 300, 0), "#55aa55", "Y"),
                                     ((0, 0, 220), "#5555cc", "Z")):
            end = project(tuple(center[i] + vector[i] for i in range(3)))
            self.canvas.create_line(*origin, *end, fill=color, width=2, arrow=tk.LAST)
            self.canvas.create_text(end[0] + 10, end[1], text=label, fill=color)

        def trail_coords(points):
            coords = []
            for point in points:
                coords.extend(project(point))
            return coords

        if len(self.driver_trail_3d) > 1:
            self.canvas.create_line(*trail_coords(self.driver_trail_3d), fill="#8aa8e8",
                                    width=2, smooth=True, capstyle=tk.ROUND)
        if len(self.runner_trail_3d) > 1:
            self.canvas.create_line(*trail_coords(self.runner_trail_3d), fill="#e9a0a0",
                                    width=2, smooth=True, capstyle=tk.ROUND)
        p = project(p3)
        e = project(e3)
        self.canvas.create_line(*p, *e, fill="#888888", dash=(5, 3), width=2)
        self.canvas.create_oval(p[0] - 12, p[1] - 12, p[0] + 12, p[1] + 12,
                                fill="blue", outline="black")
        self.canvas.create_text(p[0], p[1] - 23, text="P", fill="blue", font=("Arial", 12, "bold"))
        self.canvas.create_oval(e[0] - 10, e[1] - 10, e[0] + 10, e[1] + 10,
                                fill="red", outline="black")
        self.canvas.create_text(e[0], e[1] - 22, text="E", fill="red", font=("Arial", 12, "bold"))
        rx, ry, rz = self.model_3d.relative_position
    def loop(self):
        self.update()
        self.draw()
        self.draw_info()
        self.draw_reduced_state()
        self.root.after(int(self.dt * 1000), self.loop)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    game = Game()
    game.run()
