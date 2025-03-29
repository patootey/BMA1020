import pyglet
from pyglet.graphics import Batch
from pyglet import shapes
import numpy as np

# Window dimentions
WINDOW_HEIGHT = 1000
WINDOW_WIDTH = 1000

# Constants
CENTER = np.array([WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2])
FRAME_TIME = 1 / 60
GRAVITY_ACCELERATION = 200
PARTICLES_PER_FRAME = 5
START_PARTICLES = 100

# Global variables
g_batch = Batch()
g_window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)


class ParticleSystem:
    alive: np.ndarray
    "Array to keep track of alive particles"
    positions: np.ndarray
    "Array to keep track of all particle positions"
    velocities: np.ndarray
    "Array to keep track of all the velocites"
    radiuses: np.ndarray
    "Array to keep track of all the particle radiuses"
    shapes: np.ndarray
    "Array to keep track of all particle objects"
    batch: Batch
    "Batch of the pyglet shapes to draw in one call"
    time: float = 0
    "To keep track of time elapsed"

    def __init__(
        self,
        batch: Batch,
        alive: np.ndarray,
        shapes: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        radiuses: np.ndarray,
    ):
        """

        :param batch: Pyglet batch for drawing the shapes
        :param alive: Boolean array to track alive particles
        :param shapes: Array of Pyglet Circle shapes representing particles
        :param positions: Array of particle positions
        :param velocities: Array of particle velocities
        :param radiuses: Array of particle radiuses

        """
        self.alive = alive
        self.positions = positions
        self.velocities = velocities
        self.radiuses = radiuses
        self.shapes = shapes
        self.batch = batch

    def make_particles(self):
        """
        Creates new particles if needed and sets initial positions and velocities.
        """
        # Find dead particles
        # Got help with understanding how the where function worked
        dead_indices = np.where(~self.alive)[0][:PARTICLES_PER_FRAME]
        needed = PARTICLES_PER_FRAME - len(dead_indices)

        # If more particles are needed, append them
        # Got help with understanding that I need to append all the new paritcles made to the originals.
        if needed > 0:
            new_dead = np.arange(len(self.alive), len(self.alive) + needed)
            dead_indices = np.append(dead_indices, new_dead)

            self.alive = np.append(self.alive, np.zeros(needed, dtype=bool))
            self.positions = np.append(self.positions, np.zeros((needed, 2)), axis=0)
            self.velocities = np.append(self.velocities, np.zeros((needed, 2)), axis=0)
            radius = np.random.uniform(10, 15, needed)

            # Create new circle shapes for each new particle
            self.shapes = np.append(
                self.shapes,
                np.array(
                    [
                        shapes.Circle(
                            0,
                            0,
                            radius[i],
                            color=(
                                255,
                                255,
                                255,
                            ),
                            batch=g_batch,
                        )
                        for i in range(needed)
                    ]
                ),
            )
            self.radiuses = np.append(self.radiuses, radius)
        # Mark the dead particles as alive
        self.alive[dead_indices] = True

        # Just to get .visible comments
        shape: shapes.Circle
        for shape in self.shapes[dead_indices]:
            shape.visible = True

        # Set new positions to the center
        self.positions[dead_indices, :] = CENTER

        # Randomize initial velocities and directions
        speed = np.random.uniform(200, 300, len(dead_indices))
        angle = np.random.uniform(np.pi / 4, 3 * np.pi / 4, len(dead_indices))
        self.velocities[dead_indices, 0] = speed * np.cos(angle)
        self.velocities[dead_indices, 1] = speed * np.sin(angle)

    def update(self, dt: float):
        """
        Updates the state of the particles each frame.

        :param dt: Time passed since the last frame
        """
        self.time += dt

        # Remove particles that fall below the window (y < 0)
        make_dead = np.where(self.positions[:, 1] < 0)
        self.alive[make_dead] = False
        shape: shapes.Circle
        for shape in self.shapes[make_dead]:
            # Not draw the dead particles (when they are alive, draw again)
            shape.visible = False

        # Apply wind effect to the particles that are alive
        are_alive = np.where(self.alive)
        wind_velocity_x = np.sin(self.time) * 20
        wind_accel = (
            wind_velocity_x * 2 * self.radiuses[are_alive]
            - self.velocities[are_alive, 0]
        )
        self.velocities[are_alive, 0] += wind_accel * dt

        # Apply gravity to the particles
        self.velocities[are_alive, 1] -= GRAVITY_ACCELERATION * dt

        # Update positions based on current velocities
        self.positions[are_alive] += self.velocities[are_alive] * dt

        # Update shape positions to match particle positions
        for i, shape in enumerate(self.shapes):
            shape.x = self.positions[i, 0]
            shape.y = self.positions[i, 1]


def main() -> None:
    """
    Main function to initialize the Pyglet window and the particle system.
    """
    # Initialize particle system with initial values
    alive: np.ndarray = np.zeros(START_PARTICLES, dtype=bool)
    velocity: np.ndarray = np.zeros((START_PARTICLES, 2))
    position: np.ndarray = np.zeros((START_PARTICLES, 2))
    radius: np.ndarray = np.random.uniform(10, 15, START_PARTICLES)
    colors = np.random.randint(50, 255, (START_PARTICLES, 3))

    # Create circle shapes for each particle
    circle_shapes: np.ndarray = np.array(
        [
            shapes.Circle(
                0,
                0,
                radius[i],
                color=tuple(colors[i]),
                batch=g_batch,
            )
            for i in range(START_PARTICLES)
        ]
    )

    # Create label to display the number of alive particles
    label = pyglet.text.Label(
        "",
        font_name="Arial",
        font_size=25,
        x=100,
        y=100,
    )

    # Create ParticleSystem instance
    particles = ParticleSystem(
        batch=g_batch,
        alive=alive,
        shapes=circle_shapes,
        positions=position,
        radiuses=radius,
        velocities=velocity,
    )

    @g_window.event
    def on_draw():
        """
        Draw the particles and label on the window.
        """
        g_window.clear()
        g_batch.draw()
        label.draw()

    def update(dt: float):
        """
        Update the particle system and label.

        :param dt: Time passed since the last frame
        """
        particles.make_particles()
        particles.update(dt)
        label.text = f"{len(particles.alive)}"

    pyglet.clock.schedule_interval(update, FRAME_TIME)
    pyglet.app.run()


if __name__ == "__main__":
    main()
