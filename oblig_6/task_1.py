import pyglet
from pyglet import shapes
from pyglet.graphics import Batch
import numpy as np


# Window dimensions
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000

# Others
FRAME_TIME: float = 1 / 60
NUM_CIRCLES = 100
MIN_RADIUS = 2
MAX_RADIUS = 22
VELOCITY_RANGE = 50
COEFFICIENT_OF_RESTITUTION = 0.6

# Window
g_window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
pyglet.gl.glClearColor(0.2, 0.0, 0.1, 1)
g_batch = Batch()


class CircleSystem:
    def __init__(
        self,
        color: np.ndarray,
        shape: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: np.ndarray,
    ) -> None:
        self.positions = position
        self.velocities = velocity
        self.radiuses = radius
        self.color = color
        self.masses = self.radiuses
        self.shapes = shape

    def wall_collision(self, dt: float, window_width: int, window_height: int) -> None:
        """Update circle position and handle bouncing off walls."""
        # Calculate new position
        new_positions = self.positions + self.velocities * dt

        # Detect collisions with walls
        collision_x = (new_positions[:, 0] - self.radiuses < 0) | (
            new_positions[:, 0] + self.radiuses > window_width
        )
        collision_y = (new_positions[:, 1] - self.radiuses < 0) | (
            new_positions[:, 1] + self.radiuses > window_height
        )

        # Reverse velocities where collisions occur
        self.velocities[:, 0] *= np.where(collision_x, -1, 1)
        self.velocities[:, 1] *= np.where(collision_y, -1, 1)

        new_positions[:, 0] = np.clip(
            new_positions[:, 0], self.radiuses, window_width - self.radiuses
        )

        new_positions[:, 1] = np.clip(
            new_positions[:, 1], self.radiuses, window_height - self.radiuses
        )

        # Putting the new values to the original
        self.positions[:] = new_positions

        for i, shape in enumerate(self.shapes):
            shape.x = self.positions[i, 0]
            shape.y = self.positions[i, 1]

    def handle_collisions(self):
        for i in range(len(self.positions)):
            delta = self.positions - self.positions[i]
            distance = np.linalg.norm(delta, axis=1)
            radius_sum = self.radiuses[i] + self.radiuses
            collision_check = distance < radius_sum
            collision_check[i] = False  # Ignore self-collision
            colliding_indices = np.where(collision_check)[0]
            if len(colliding_indices) > 0:
                self.circle_collision(
                    np.array([i] * len(colliding_indices)), colliding_indices
                )

    def circle_collision(self, circle1: np.ndarray, circle2: np.ndarray) -> None:
        # Calculate distance between circle centers
        normal = self.positions[circle2] - self.positions[circle1]

        # normalizing
        distance = np.linalg.norm(normal)

        # Avoid division by zero
        if distance == 0:
            return

        normal /= distance

        # Calculate normal velocities
        vec1_normal = np.dot(self.velocities[circle1], normal)
        vec2_normal = np.dot(self.velocities[circle2], normal)

        mass = (self.masses[circle1] * self.masses[circle2]) / (
            self.masses[circle1] + self.masses[circle2]
        )

        # Calculate new normal velocities using impulse
        new_normal1 = vec1_normal + (mass / self.masses[circle1]) * (
            1 + COEFFICIENT_OF_RESTITUTION
        ) * (vec2_normal - vec1_normal)
        new_normal2 = vec2_normal - (mass / self.masses[circle2]) * (
            1 + COEFFICIENT_OF_RESTITUTION
        ) * (vec2_normal - vec1_normal)

        # Calculate tangential velocities
        v1_tangent = self.velocities[circle1] - vec1_normal * normal
        v2_tangent = self.velocities[circle2] - vec2_normal * normal

        # Add to the velocity, so show the direction
        self.velocities[circle1] = new_normal1 * normal + v1_tangent
        self.velocities[circle2] = new_normal2 * normal + v2_tangent

        # Resolve overlap by separating circles
        overlap = self.radiuses[circle1] + self.radiuses[circle2] - distance
        total_mass = self.masses[circle1] + self.masses[circle2]

        # Displace circles proportionally to their masses
        self.positions[circle1] -= (
            overlap * (self.masses[circle2] / total_mass) * normal
        )
        self.positions[circle2] += (
            overlap * (self.masses[circle1] / total_mass) * normal
        )


def main() -> None:
    radiuses = np.random.uniform(MIN_RADIUS, MAX_RADIUS, NUM_CIRCLES)
    colors = np.random.randint(50, 255, (NUM_CIRCLES, 3))
    positions = np.random.uniform(
        MAX_RADIUS, WINDOW_WIDTH - MAX_RADIUS, (NUM_CIRCLES, 2)
    )
    velocities = np.random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE, (NUM_CIRCLES, 2))
    circle_shapes = np.array(
        [
            shapes.Circle(0, 0, radiuses[i], color=tuple(colors[i]), batch=g_batch)
            for i in range(NUM_CIRCLES)
        ]
    )

    circles = CircleSystem(
        position=positions,
        velocity=velocities,
        radius=radiuses,
        color=colors,
        shape=circle_shapes,
    )

    def update(dt) -> None:
        """Update all circles and handle collisions."""

        circles.handle_collisions()
        circles.wall_collision(dt, WINDOW_WIDTH, WINDOW_HEIGHT)

    @g_window.event
    def on_draw() -> None:
        """Draw all elements on the screen."""
        g_window.clear()
        g_batch.draw()

    # Run update function every frame
    pyglet.clock.schedule_interval(update, FRAME_TIME)
    pyglet.app.run()


if __name__ == "__main__":
    main()
