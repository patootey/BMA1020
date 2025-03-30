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
g_window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
pyglet.gl.glClearColor(0.2, 0.0, 0.1, 1)


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
        colliding_circle_a = []
        colliding_circle_b = []
        for circle in range(NUM_CIRCLES - 1):
            # Did circle + 1 to stop self collding checks and skip circles already collided
            delta = self.positions[circle + 1 :] - self.positions[circle]
            distance = np.linalg.norm(delta, axis=1)
            radius_sum = self.radiuses[circle] + self.radiuses[circle + 1 :]
            collision_check = distance < radius_sum

            colliding_indices = np.where(collision_check)[0] + circle + 1

            # Saving which circles are colliding
            colliding_circle_a.extend(colliding_indices)
            colliding_circle_b.extend([circle] * len(colliding_indices))

        # Call the function one time for all colliding circles
        if len(colliding_circle_a) > 0:
            self.circle_collision(
                np.array(colliding_circle_a), np.array(colliding_circle_b)
            )

    def circle_collision(self, circle1: np.ndarray, circle2: np.ndarray) -> None:
        """
        Handle the collision between two circles.

        :param circle1: The first circle array to collide
        :param circle2: The second circle array to collide
        """
        # Calculate distance between circle centers
        vectors = self.positions[circle2] - self.positions[circle1]

        # Finding length of all vectors
        distances = np.linalg.norm(vectors, axis=1)

        # Avoid dividing by zero (finding which are not zero)
        non_zero = np.where(distances != 0)[0]

        # Removes elementes where distance is zero
        vectors = vectors[non_zero]
        distances = distances[non_zero]
        circle1 = circle1[non_zero]
        circle2 = circle2[non_zero]

        # If there is no element, there is nothing to do
        if len(non_zero) == 0:
            return

        # Newaxis for mulipling/dividing by scalars (convert to 2D)
        normalized_vectors = vectors / distances[:, np.newaxis]

        # Calculate normal velocities
        vec1_normals = np.vecdot(self.velocities[circle1], normalized_vectors)
        vec2_normals = np.vecdot(self.velocities[circle2], normalized_vectors)

        masses = (self.masses[circle1] * self.masses[circle2]) / (
            self.masses[circle1] + self.masses[circle2]
        )

        # Calculate new normal velocities using impulse
        impulse_normals1 = vec1_normals + (masses / self.masses[circle1]) * (
            1 + COEFFICIENT_OF_RESTITUTION
        ) * (vec2_normals - vec1_normals)
        impulse_normals2 = vec2_normals - (masses / self.masses[circle2]) * (
            1 + COEFFICIENT_OF_RESTITUTION
        ) * (vec2_normals - vec1_normals)

        # Calculate tangential velocities
        v1_tangents = (
            self.velocities[circle1] - vec1_normals[:, np.newaxis] * normalized_vectors
        )
        v2_tangents = (
            self.velocities[circle2] - vec2_normals[:, np.newaxis] * normalized_vectors
        )

        # Add to the velocity, so show the direction
        self.velocities[circle1] = (
            impulse_normals1[:, np.newaxis] * normalized_vectors
        ) + v1_tangents
        self.velocities[circle2] = (
            impulse_normals2[:, np.newaxis] * normalized_vectors + v2_tangents
        )

        # Resolve overlap by separating circles
        overlaps = self.radiuses[circle1] + self.radiuses[circle2] - distances
        total_masses = self.masses[circle1] + self.masses[circle2]

        # Displace circles proportionally to their masses
        self.positions[circle1] -= (overlaps * self.masses[circle2] / total_masses)[
            :, np.newaxis
        ] * normalized_vectors
        self.positions[circle2] += (overlaps * self.masses[circle1] / total_masses)[
            :, np.newaxis
        ] * normalized_vectors


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
