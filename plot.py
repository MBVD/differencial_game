import numpy as np
import matplotlib.pyplot as plt


class ValueFunctionPlotter:
    def __init__(self, x_values, y_values, value_grid, capture_radius):
        self.x_values = np.array(x_values, dtype=float)
        self.y_values = np.array(y_values, dtype=float)
        self.value_grid = np.array(value_grid, dtype=float)
        self.capture_radius = capture_radius

        self.X, self.Y = np.meshgrid(self.x_values, self.y_values)

    def plot_surface(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            self.X,
            self.Y,
            self.value_grid,
            linewidth=0,
            antialiased=True
        )

        ax.set_title("Функция платы V(x, y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("V(x, y)")

        plt.tight_layout()
        plt.show()

    def plot_contour(self):
        plt.figure(figsize=(9, 7))

        contour = plt.contour(
            self.X,
            self.Y,
            self.value_grid,
            levels=20
        )

        plt.clabel(contour, inline=True, fontsize=8)

        circle = plt.Circle(
            (0, 0),
            self.capture_radius,
            fill=False,
            linewidth=2
        )
        plt.gca().add_patch(circle)

        plt.title("Линии уровня функции платы V(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="V(x, y)")
        plt.gca().set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()

    def plot_heatmap(self):
        plt.figure(figsize=(9, 7))

        plt.imshow(
            self.value_grid,
            origin="lower",
            extent=[
                self.x_values.min(),
                self.x_values.max(),
                self.y_values.min(),
                self.y_values.max()
            ],
            aspect="equal"
        )

        circle = plt.Circle(
            (0, 0),
            self.capture_radius,
            fill=False,
            linewidth=2
        )
        plt.gca().add_patch(circle)

        plt.title("Тепловая карта функции платы V(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="V(x, y)")

        plt.tight_layout()
        plt.show()

    def plot_contour_zoom(self, x_min=-120.0, x_max=120.0, y_min=-120.0, y_max=120.0):
        mask_x = (self.x_values >= x_min) & (self.x_values <= x_max)
        mask_y = (self.y_values >= y_min) & (self.y_values <= y_max)

        x_zoom = self.x_values[mask_x]
        y_zoom = self.y_values[mask_y]
        value_zoom = self.value_grid[np.ix_(mask_y, mask_x)]

        X_zoom, Y_zoom = np.meshgrid(x_zoom, y_zoom)

        plt.figure(figsize=(9, 7))

        contour = plt.contour(
            X_zoom,
            Y_zoom,
            value_zoom,
            levels=20
        )

        plt.clabel(contour, inline=True, fontsize=8)

        circle = plt.Circle(
            (0, 0),
            self.capture_radius,
            fill=False,
            linewidth=2
        )
        plt.gca().add_patch(circle)

        plt.title("Функция платы V(x, y) около области захвата")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="V(x, y)")
        plt.gca().set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()

    def save_surface(self, filename="V_surface.png"):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            self.X,
            self.Y,
            self.value_grid,
            linewidth=0,
            antialiased=True
        )

        ax.set_title("Функция платы V(x, y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("V(x, y)")

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close(fig)

    def save_contour(self, filename="V_contour.png"):
        plt.figure(figsize=(9, 7))

        contour = plt.contour(
            self.X,
            self.Y,
            self.value_grid,
            levels=20
        )

        plt.clabel(contour, inline=True, fontsize=8)

        circle = plt.Circle(
            (0, 0),
            self.capture_radius,
            fill=False,
            linewidth=2
        )
        plt.gca().add_patch(circle)

        plt.title("Линии уровня функции платы V(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="V(x, y)")
        plt.gca().set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

    def save_heatmap(self, filename="V_heatmap.png"):
        plt.figure(figsize=(9, 7))

        plt.imshow(
            self.value_grid,
            origin="lower",
            extent=[
                self.x_values.min(),
                self.x_values.max(),
                self.y_values.min(),
                self.y_values.max()
            ],
            aspect="equal"
        )

        circle = plt.Circle(
            (0, 0),
            self.capture_radius,
            fill=False,
            linewidth=2
        )
        plt.gca().add_patch(circle)

        plt.title("Тепловая карта функции платы V(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="V(x, y)")

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

    def save_contour_zoom(
        self,
        filename="V_contour_zoom.png",
        x_min=-120.0,
        x_max=120.0,
        y_min=-120.0,
        y_max=120.0
    ):
        mask_x = (self.x_values >= x_min) & (self.x_values <= x_max)
        mask_y = (self.y_values >= y_min) & (self.y_values <= y_max)

        x_zoom = self.x_values[mask_x]
        y_zoom = self.y_values[mask_y]
        value_zoom = self.value_grid[np.ix_(mask_y, mask_x)]

        X_zoom, Y_zoom = np.meshgrid(x_zoom, y_zoom)

        plt.figure(figsize=(9, 7))

        contour = plt.contour(
            X_zoom,
            Y_zoom,
            value_zoom,
            levels=20
        )

        plt.clabel(contour, inline=True, fontsize=8)

        circle = plt.Circle(
            (0, 0),
            self.capture_radius,
            fill=False,
            linewidth=2
        )
        plt.gca().add_patch(circle)

        plt.title("Функция платы V(x, y) около области захвата")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="V(x, y)")
        plt.gca().set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

    def save_all(self):
        self.save_surface("V_surface.png")
        self.save_contour("V_contour.png")
        self.save_heatmap("V_heatmap.png")
        self.save_contour_zoom("V_contour_zoom.png")

        print("Графики функции платы сохранены:")
        print("V_surface.png")
        print("V_contour.png")
        print("V_heatmap.png")
        print("V_contour_zoom.png")