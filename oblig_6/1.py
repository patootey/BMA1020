import pyglet
from pyglet import shapes
from pyglet.math import Vec2
from pyglet.graphics import Batch
import random
import math
import numpy as np

# IT WOOOOOOOOOOOOOOOOOOOORKS, bytt til numpy

# Window dimensions
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
# Others
FRAME_TIME: float = 1 / 60
NUM_CIRCLES = 100
MIN_RADIUS = 2
MAX_RADIUS = 22
VELOCITY_RANGE = 50
COEFFICIENT_OF_RESTITUTION = (
    0.6  # How "bouncy" collisions are (1.0 = perfectly elastic)
)

# Window
g_window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
g_batch = Batch()


# General functions
def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between min_value and max_value."""
    return max(min(value, max_value), min_value)


def distance_between_points(point1: Vec2, point2: Vec2) -> float:
    """Calculate the distance between two points."""
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)


# Circle class: defines a circle with a center point and a radius
class Circle:
    def __init__(
        self,
        batch: Batch,
        color: tuple[int, int, int],
        pos: Vec2 | None = None,
        velocity: Vec2 | None = None,
        radius: float = 1,
    ) -> None:
        self.pos = pos or Vec2()
        self.velocity = velocity or Vec2()
        self.radius = radius
        self.color = color
        self.mass = self.radius

        # The circle object drawn on the screen
        self.shape = shapes.Circle(
            self.pos.x,
            self.pos.y,
            self.radius,
            color=self.color,
            batch=batch,
        )

    def check_collision(self, other) -> bool:
        """Check if this circle is colliding with another circle."""
        return distance_between_points(self.pos, other.pos) < (
            self.radius + other.radius
        )

    def update(self, dt: float, window_width: int, window_height: int) -> None:
        """Update circle position and handle bouncing off walls."""
        new_pos: Vec2 = self.pos + self.velocity * dt  # Calculate new position

        # Bounce off left or right walls
        if new_pos.x - self.radius < 0 or new_pos.x + self.radius > window_width:
            self.velocity = Vec2(-self.velocity.x, self.velocity.y)
            new_pos = Vec2(
                clamp(new_pos.x, self.radius, window_width - self.radius), new_pos.y
            )

        # Bounce off top or bottom walls
        if new_pos.y - self.radius < 0 or new_pos.y + self.radius > window_height:
            self.velocity = Vec2(self.velocity.x, -self.velocity.y)
            new_pos = Vec2(
                new_pos.x, clamp(new_pos.y, self.radius, window_height - self.radius)
            )

        # Putting the new values to the original
        self.pos = new_pos

        # Update shape position
        self.shape.x = self.pos.x
        self.shape.y = self.pos.y

    def resolve_collision(self, other: "Circle") -> None:
        """
        Resolve collision between two circles using coefficient of restitution.
        This simulates physically accurate bouncing between objects with energy loss.
        """
        # Get positions as numpy arrays for vector calculations
        pos1 = np.array([self.pos.x, self.pos.y], dtype=float)
        pos2 = np.array([other.pos.x, other.pos.y], dtype=float)

        # Calculate distance between circles
        separation_distance = distance_between_points(self.pos, other.pos)

        # Avoid division by zero
        if separation_distance == 0:
            return

        # Calculate normal vector (direction of collision)
        normal_vec = (pos2 - pos1) / separation_distance

        # Convert velocities to numpy arrays
        vel1 = np.array([self.velocity.x, self.velocity.y], dtype=float)
        vel2 = np.array([other.velocity.x, other.velocity.y], dtype=float)

        # Calculate the velocities along the normal direction
        v1_normal = np.dot(vel1, normal_vec)
        v2_normal = np.dot(vel2, normal_vec)

        # Skip collision handling if objects are moving away from each other
        if v1_normal < v2_normal:
            return

        # Calculate masses
        m1 = self.mass
        m2 = other.mass

        # Calculate new velocities along normal using coefficient of restitution
        # Conservation of momentum and energy with restitution coefficient
        v1_normal_new = (
            (m1 - COEFFICIENT_OF_RESTITUTION * m2) * v1_normal
            + (1 + COEFFICIENT_OF_RESTITUTION) * m2 * v2_normal
        ) / (m1 + m2)

        v2_normal_new = (
            (m2 - COEFFICIENT_OF_RESTITUTION * m1) * v2_normal
            + (1 + COEFFICIENT_OF_RESTITUTION) * m1 * v1_normal
        ) / (m1 + m2)

        # Convert normal velocities back to vectors
        v1_normal_vec = v1_normal_new * normal_vec
        v2_normal_vec = v2_normal_new * normal_vec

        # Calculate tangential component (velocity not affected by collision)
        tangent_vec = np.array([-normal_vec[1], normal_vec[0]], dtype=float)
        v1_tangent = np.dot(vel1, tangent_vec) * tangent_vec
        v2_tangent = np.dot(vel2, tangent_vec) * tangent_vec

        # Set new velocities (normal + tangential components)
        new_vel1 = v1_normal_vec + v1_tangent
        new_vel2 = v2_normal_vec + v2_tangent

        # Update velocities in original format
        self.velocity = Vec2(new_vel1[0], new_vel1[1])
        other.velocity = Vec2(new_vel2[0], new_vel2[1])

        # Move circles apart to prevent sticking (resolve overlap)
        overlap = self.radius + other.radius - separation_distance

        # Adjust positions proportionally to masses to prevent overlap
        total_mass = m1 + m2
        self_ratio = m2 / total_mass
        other_ratio = m1 / total_mass

        self.pos = Vec2(
            self.pos.x - normal_vec[0] * overlap * self_ratio,
            self.pos.y - normal_vec[1] * overlap * self_ratio,
        )

        other.pos = Vec2(
            other.pos.x + normal_vec[0] * overlap * other_ratio,
            other.pos.y + normal_vec[1] * overlap * other_ratio,
        )


def main() -> None:
    # Create a list to store the circles
    circles = []

    # Set up NUM_CIRCLES with random positions, velocities and sizes
    for i in range(NUM_CIRCLES):
        circles.append(
            Circle(
                pos=Vec2(
                    random.uniform(MAX_RADIUS, WINDOW_WIDTH - MAX_RADIUS),
                    random.uniform(MAX_RADIUS, WINDOW_HEIGHT - MAX_RADIUS),
                ),
                velocity=Vec2(
                    random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE),
                    random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE),
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
            circle.update(dt, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Collision detection and resolution
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                if circles[i].check_collision(circles[j]):
                    circles[i].resolve_collision(circles[j])

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
