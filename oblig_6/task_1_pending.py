import pyglet
from pyglet import shapes
from pyglet.graphics import Batch
import numpy as np


# Window dimensions
WINDOW_HEIGHT = 1000
WINDOW_WIDTH = 1000

# Constants
COEFFICIENT_OF_RESTITUTION = 0.6
FRAME_TIME = 1 / 60
MAX_RADIUS = 22
MIN_RADIUS = 2
NUM_CIRCLES = 100
VELOCITY_RANGE = 50

# Global variables
g_batch = Batch()
pyglet.gl.glClearColor(0.2, 0.0, 0.1, 1)
g_window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)


class CircleSystem:
    positions: np.ndarray
    "Array of all circle positions"
    velocities: np.ndarray
    "Array of all circle velocites"
    radiuses: np.ndarray
    "Array of all circle radiuses"
    color: np.ndarray
    "Array of random colors for all circle objects"
    masses: np.ndarray
    "Array of all circle masses"
    shapes: np.ndarray
    "Array of all circle shape objects"

    def __init__(
        self,
        color: np.ndarray,
        shapes: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        radiuses: np.ndarray,
    ) -> None:
        """
        Class to handle the physics and state of all circles in the system.

        :param color: Random colors for each circle
        :param shapes: Pyglet shapes (Circle objects) for each circle
        :param positions: Initial positions of the circles
        :param velocities: Initial velocities of the circles
        :param radiuses: Radiuses of the circles
        """
        self.positions = positions
        self.velocities = velocities
        self.radiuses = radiuses
        self.color = color
        self.masses = self.radiuses
        self.shapes = shapes

    def wall_collision(self, dt: float, window_width: int, window_height: int) -> None:
        """Update circle positions and handle collisions with the window boundaries."""
        # Calculate new position
        new_positions = self.positions + self.velocities * dt

        # Detect collisions with the window walls (left/right and top/bottom)
        collision_x = (new_positions[:, 0] - self.radiuses < 0) | (
            new_positions[:, 0] + self.radiuses > window_width
        )
        collision_y = (new_positions[:, 1] - self.radiuses < 0) | (
            new_positions[:, 1] + self.radiuses > window_height
        )

        # Reverse the velocities in the x and y directions where collisions occur
        # Got help with the knowledge og np.where from my father
        self.velocities[:, 0] *= np.where(collision_x, -1, 1)
        self.velocities[:, 1] *= np.where(collision_y, -1, 1)

        # Clip(clamp) the new positions to make sure the circles stay within the window bounds
        # Switched to clip beacuse its the same as clamp
        new_positions[:, 0] = np.clip(
            new_positions[:, 0], self.radiuses, window_width - self.radiuses
        )

        new_positions[:, 1] = np.clip(
            new_positions[:, 1], self.radiuses, window_height - self.radiuses
        )

        # Update the positions to the new clipped positions
        self.positions[:] = new_positions

        # Update the Pyglet shapes (circles) to match the new positions
        for i, shape in enumerate(self.shapes):
            shape.x = self.positions[i, 0]
            shape.y = self.positions[i, 1]

    def handle_collisions(self):
        """Handle collisions between all circles in the system."""
        for circle in range(NUM_CIRCLES):
            # Calculate the difference between the current circle and all other circles
            delta = self.positions - self.positions[circle]
            distance = np.linalg.norm(delta, axis=1)

            # Determine which circles are colliding
            radius_sum = self.radiuses[circle] + self.radiuses
            collision_check = distance < radius_sum

            # Find the indices of all circles that are colliding with the current circle
            colliding_indices = np.where(collision_check)[0]

            # Handle collisions with all the colliding circles
            for circle_2 in colliding_indices:
                self.circle_collision(circle, circle_2)

    def circle_collision(self, circle1, circle2) -> None:
        """
        Handle the collision between two circles.

        :param circle1: The first circle to collide
        :param circle2: The second circle to collide
        """
        # Calculate distance between circle centers
        normal = self.positions[circle2] - self.positions[circle1]

        # Calculate the distance between the two circles
        distance = np.linalg.norm(normal)

        # Avoid division by zero
        if distance == 0:
            return

        # Normalize the normal vector
        normal /= distance

        # Calculate normal velocities
        vec1_normal = np.dot(self.velocities[circle1], normal)
        vec2_normal = np.dot(self.velocities[circle2], normal)

        # Calculate the reduced mass for the collision
        mass = (self.masses[circle1] * self.masses[circle2]) / (
            self.masses[circle1] + self.masses[circle2]
        )

        # Calculate new velocities in the normal direction using the impulse formula (charles)
        new_normal1 = vec1_normal + (mass / self.masses[circle1]) * (
            1 + COEFFICIENT_OF_RESTITUTION
        ) * (vec2_normal - vec1_normal)
        new_normal2 = vec2_normal - (mass / self.masses[circle2]) * (
            1 + COEFFICIENT_OF_RESTITUTION
        ) * (vec2_normal - vec1_normal)

        # Calculate tangential velocities
        v1_tangent = self.velocities[circle1] - vec1_normal * normal
        v2_tangent = self.velocities[circle2] - vec2_normal * normal

        # Update the velocities by combining normal and tangential components
        self.velocities[circle1] = new_normal1 * normal + v1_tangent
        self.velocities[circle2] = new_normal2 * normal + v2_tangent

        # Resolve the overlap between the circles by moving them apart
        # Got some help from my father to understand when it has a overlap
        overlap = self.radiuses[circle1] + self.radiuses[circle2] - distance
        total_mass = self.masses[circle1] + self.masses[circle2]

        # Displace the circles proportionally to their masses
        self.positions[circle1] -= (
            overlap * (self.masses[circle2] / total_mass) * normal
        )
        self.positions[circle2] += (
            overlap * (self.masses[circle1] / total_mass) * normal
        )


def main() -> None:
    """Main function to initialize the circles and run the simulation."""
    # Randomly generate radiuses, colors, positions, and velocities for all circles
    radiuses = np.random.uniform(MIN_RADIUS, MAX_RADIUS, NUM_CIRCLES)
    colors = np.random.randint(50, 255, (NUM_CIRCLES, 3))
    positions = np.random.uniform(
        MAX_RADIUS, WINDOW_WIDTH - MAX_RADIUS, (NUM_CIRCLES, 2)
    )
    velocities = np.random.uniform(-VELOCITY_RANGE, VELOCITY_RANGE, (NUM_CIRCLES, 2))

    # Create Pyglet circle shapes for each circle
    circle_shapes = np.array(
        [
            shapes.Circle(0, 0, radiuses[i], color=tuple(colors[i]), batch=g_batch)
            for i in range(NUM_CIRCLES)
        ]
    )

    # Initialize the CircleSystem
    circles = CircleSystem(
        positions=positions,
        velocities=velocities,
        radiuses=radiuses,
        color=colors,
        shapes=circle_shapes,
    )

    def update(dt: float) -> None:
        """
        Update all circles and handle collisions and wall collisions.

        :param dt: Time passed since the last frame
        """

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
