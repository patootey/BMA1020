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

# bytte om til en mer effektiv kode
# hør med TA, ta sa det var greit?


# General functions
def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between min_value and max_value."""
    return max(min(value, max_value), min_value)


def distance_between_points(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate the distance between two points."""
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


g_position = np.zeros((NUM_CIRCLES, 2))
g_velocity = np.zeros((NUM_CIRCLES, 2))
g_radius = np.random.uniform(MIN_RADIUS, MAX_RADIUS, NUM_CIRCLES)
g_color: tuple[int, int, int] = (
    np.random.randint(50, 255),
    np.random.randint(50, 255),
    np.random.randint(50, 255),
)


# Circle class: defines a circle with a center point and a radius
class CircleSystem:
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
        # Calculate new position
        new_pos = self.position + self.velocity * dt

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

        # bruke linalg? syntes det hakket
        # r = other.position - self.position
        distance = distance_between_points(self.position, other.position)
        # distance = np.linalg.norm(,r)
        # Avoid division by zero
        if distance == 0:
            return

        # Calculate collision normal
        normalize = (other.position - self.position) / distance
        # Calculate normal velocities
        vec1_normal = np.dot(self.velocity, normalize)
        vec2_normal = np.dot(other.velocity, normalize)

        mass = (self.mass * other.mass) / (self.mass + other.mass)

        # Calculate new normal velocities using impulse
        Impulse1 = vec1_normal + (mass / self.mass) * (
            1 + COEFFICIENT_OF_RESTITUTION
        ) * (vec2_normal - vec1_normal)
        Impulse2 = vec2_normal - (mass / other.mass) * (
            1 + COEFFICIENT_OF_RESTITUTION
        ) * (vec2_normal - vec1_normal)

        # Calculate tangential velocities
        v1_tangent = self.velocity - vec1_normal * normalize
        v2_tangent = other.velocity - vec2_normal * normalize

        # Add to the velocity, so show the direction
        self.velocity = Impulse1 * normalize + v1_tangent
        other.velocity = Impulse2 * normalize + v2_tangent

        # Resolve overlap by separating circles
        overlap = self.radius + other.radius - distance
        total_mass = self.mass + other.mass

        # Displace circles proportionally to their masses
        self.position -= overlap * (other.mass / total_mass) * normalize
        other.position += overlap * (self.mass / total_mass) * normalize


def main() -> None:
    circles = []

    for i in range(NUM_CIRCLES):
        circles.append(
            Circle(
                position=np.array(
                    [
                        np.random.uniform(MAX_RADIUS, WINDOW_WIDTH - MAX_RADIUS),
                        np.random.uniform(MAX_RADIUS, WINDOW_HEIGHT - MAX_RADIUS),
                    ]
                ),
                velocity=np.array(
                    [
                        np.random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE),
                        np.random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE),
                    ]
                ),
                radius=np.random.randint(MIN_RADIUS, MAX_RADIUS),
                color=(
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
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
