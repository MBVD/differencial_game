"""A small 3-D simple-motion evader / bounded-turn pursuer model.

The state is integrated in world coordinates and also exposes the three
relative coordinates in the pursuer's moving frame. No third-party packages
are needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vector3 = tuple[float, float, float]


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@dataclass
class Pursuit3D:
    w1: float = 120.0
    w2: float = 100.0
    turn_radius: float = 120.0
    capture_radius: float = 35.0
    pursuer_position: Vector3 = (300.0, 500.0, 0.0)
    evader_position: Vector3 = (600.0, 250.0, 80.0)
    yaw: float = 0.0
    pitch: float = 0.0
    evader_azimuth: float = math.pi / 4.0
    evader_elevation: float = 0.0
    elapsed: float = 0.0
    captured: bool = False

    @property
    def forward(self) -> Vector3:
        cp = math.cos(self.pitch)
        return cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)

    @property
    def right(self) -> Vector3:
        return -math.sin(self.yaw), math.cos(self.yaw), 0.0

    @property
    def up(self) -> Vector3:
        sp, cp = math.sin(self.pitch), math.cos(self.pitch)
        return -sp * math.cos(self.yaw), -sp * math.sin(self.yaw), cp

    @property
    def relative_position(self) -> Vector3:
        delta = tuple(self.evader_position[i] - self.pursuer_position[i] for i in range(3))
        return _dot(delta, self.forward), _dot(delta, self.right), _dot(delta, self.up)

    @property
    def separation(self) -> float:
        return math.sqrt(sum((self.evader_position[i] - self.pursuer_position[i]) ** 2
                             for i in range(3)))

    def step(
        self,
        yaw_control: float,
        pitch_control: float,
        evader_azimuth_control: float,
        evader_elevation_control: float,
        dt: float,
    ) -> None:
        """Advance with normalized pursuer turn and evader sphere-direction controls."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        if self.captured:
            return
        turn_rate = self.w1 / self.turn_radius
        magnitude = math.hypot(yaw_control, pitch_control)
        if magnitude > 1.0:
            yaw_control /= magnitude
            pitch_control /= magnitude

        # Yaw and pitch form orthogonal components of a bounded angular rate.
        self.yaw += turn_rate * yaw_control * dt
        self.pitch = max(-math.pi / 2.0, min(math.pi / 2.0,
                                             self.pitch + turn_rate * pitch_control * dt))

        self.evader_azimuth += evader_azimuth_control * dt
        self.evader_elevation = max(-math.pi / 2.0, min(math.pi / 2.0,
            self.evader_elevation + evader_elevation_control * dt))

        f = self.forward
        ca = math.cos(self.evader_elevation)
        evader_direction = (ca * math.cos(self.evader_azimuth),
                            ca * math.sin(self.evader_azimuth),
                            math.sin(self.evader_elevation))
        self.pursuer_position = tuple(self.pursuer_position[i] + self.w1 * f[i] * dt
                                      for i in range(3))
        self.evader_position = tuple(self.evader_position[i] + self.w2 * evader_direction[i] * dt
                                     for i in range(3))
        self.elapsed += dt
        self.captured = self.separation <= self.capture_radius


def relative_dynamics(
    relative_position: Vector3,
    pursuer_angular_velocity_body: Vector3,
    evader_direction_body: Vector3,
    w1: float,
    w2: float,
) -> Vector3:
    """Evaluate r_dot = -omega x r - w1 e_forward + w2 u in body coordinates."""
    wx, wy, wz = pursuer_angular_velocity_body
    rx, ry, rz = relative_position
    ux, uy, uz = evader_direction_body
    return (
        -(wy * rz - wz * ry) - w1 + w2 * ux,
        -(wz * rx - wx * rz) + w2 * uy,
        -(wx * ry - wy * rx) + w2 * uz,
    )
