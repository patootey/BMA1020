import pyglet
from pyglet import shapes
from pyglet.graphics import Batch
import random
import math
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
COEFFICIENT_OF_RESTITUTION = 0.3

# Window
g_window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
pyglet.gl.glClearColor(0.2, 0.0, 0.1, 1)
g_batch = Batch()

# bytte om til en mer effektiv kode
# hør med TA


# General functions
def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between min_value and max_value."""
    return max(min(value, max_value), min_value)


def distance_between_points(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate the distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


# Circle class: defines a circle with a center point and a radius
class Circle:
    def __init__(
        self,
        batch: Batch,
        color: tuple[int, int, int],
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float = 1,
    ) -> None:
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.color = color
        self.mass = self.radius

        # The circle object drawn on the screen
        self.shape = shapes.Circle(
            self.position[0],
            self.position[1],
            self.radius,
            color=self.color,
            batch=batch,
        )

    def check_collision(self, other) -> bool:
        """Check if this circle is colliding with another circle."""
        return distance_between_points(self.position, other.position) < (
            self.radius + other.radius
        )

    def wall_collision(self, dt: float, window_width: int, window_height: int) -> None:
        """Update circle position and handle bouncing off walls."""
        new_pos = self.position + self.velocity * dt  # Calculate new position

        # Bounce off left or right walls
        if new_pos[0] - self.radius < 0 or new_pos[0] + self.radius > window_width:
            self.velocity = np.array([-self.velocity[0], self.velocity[1]])
            new_pos[0] = clamp(new_pos[0], self.radius, window_width - self.radius)

        # Bounce off top or bottom walls
        if new_pos[1] - self.radius < 0 or new_pos[1] + self.radius > window_height:
            self.velocity = np.array([self.velocity[0], -self.velocity[1]])
            new_pos[1] = clamp(new_pos[1], self.radius, window_height - self.radius)

        # Putting the new values to the original
        self.position = new_pos

        # Update shape position
        self.shape.x = self.position[0]
        self.shape.y = self.position[1]

    def circle_collision(self, other: "Circle") -> None:
        # Calculate distance between circle centers
        distance = distance_between_points(self.position, other.position)

        # Avoid division by zero
        if distance == 0:
            return

        # Calculate collision normal (unit vector pointing from self to other)
        collision_normalized = (other.position - self.position) / distance

        # Project velocities onto the collision normal
        self_velocity_along_normal = np.dot(self.velocity, collision_normalized)
        other_velocity_along_normal = np.dot(other.velocity, collision_normalized)

        # Skip if circles are moving away from each other
        if self_velocity_along_normal < other_velocity_along_normal:
            return

        # Calculate post-collision velocities along normal using conservation of momentum
        # and coefficient of restitution
        # hjelp av
        self_new_velocity_along_normal = (
            (self.mass - COEFFICIENT_OF_RESTITUTION * other.mass)
            * self_velocity_along_normal
            + (1 + COEFFICIENT_OF_RESTITUTION)
            * other.mass
            * other_velocity_along_normal
        ) / (self.mass + other.mass)

        other_new_velocity_along_normal = (
            (other.mass - COEFFICIENT_OF_RESTITUTION * self.mass)
            * other_velocity_along_normal
            + (1 + COEFFICIENT_OF_RESTITUTION) * self.mass * self_velocity_along_normal
        ) / (self.mass + other.mass)

        # Convert scalar normal velocities back to vectors
        self_normal_velocity_vector = (
            self_new_velocity_along_normal * collision_normalized
        )
        other_normal_velocity_vector = (
            other_new_velocity_along_normal * collision_normalized
        )

        # Calculate tangential component
        collision_tangent = np.array(
            [-collision_normalized[1], collision_normalized[0]], dtype=float
        )

        # Project original velocities onto tangential direction
        self_tangential_velocity = (
            np.dot(self.velocity, collision_tangent) * collision_tangent
        )
        other_tangential_velocity = (
            np.dot(other.velocity, collision_tangent) * collision_tangent
        )

        self.velocity = self_normal_velocity_vector + self_tangential_velocity
        other.velocity = other_normal_velocity_vector + other_tangential_velocity

        # Resolve overlapping circles by moving them apart
        overlap_distance = self.radius + other.radius - distance

        # Distribute displacement based on mass ratio
        total_mass = self.mass + other.mass
        self_displacement_ratio = other.mass / total_mass
        other_displacement_ratio = self.mass / total_mass

        self.position = np.array(
            [
                self.position[0]
                - collision_normalized[0] * overlap_distance * self_displacement_ratio,
                self.position[1]
                - collision_normalized[1] * overlap_distance * self_displacement_ratio,
            ]
        )

        other.position = np.array(
            [
                other.position[0]
                + collision_normalized[0] * overlap_distance * other_displacement_ratio,
                other.position[1]
                + collision_normalized[1] * overlap_distance * other_displacement_ratio,
            ]
        )


def main() -> None:
    circles = []

    for i in range(NUM_CIRCLES):
        circles.append(
            Circle(
                position=np.array(
                    [
                        random.uniform(MAX_RADIUS, WINDOW_WIDTH - MAX_RADIUS),
                        random.uniform(MAX_RADIUS, WINDOW_HEIGHT - MAX_RADIUS),
                    ]
                ),
                velocity=np.array(
                    [
                        random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE),
                        random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE),
                    ]
                ),
                radius=random.randint(MIN_RADIUS, MAX_RADIUS),
                color=(
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255),
                ),
                batch=g_batch,
            )
        )

    def update(dt) -> None:
        """Update all circles and handle collisions."""
        for circle in circles:
            circle.wall_collision(dt, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Collision detection and resolution
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                if circles[i].check_collision(circles[j]):
                    circles[i].circle_collision(circles[j])

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
