import pyglet
from pyglet.graphics import Batch
import numpy as np

# Constants
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
START_PARTICLES = 100
GRAVITY_ACCELERATION = 200
CENTER = np.array([WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2])
PARTICLES_PER_FRAME = 5
FRAME_TIME = 1 / 60

# alltid legge til 5 per sekund

# bruk istedenfor
g_batch = Batch()
g_window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

# kanskje ikke?, hvis det er greit med funksjoner gidder jeg ikk
# numpy array, om de er i live
# numpy velocity
# numpy posisjon
# array med shapes
# g_alive = np.array([])
# g_velocity = np.array([])
# g_position = np.array([])
# g_shapes = np.array([])
# g_radius = np.array([])


# gjør til funksjoner, sende inn particel system , med index


# MANGLER 5 PARTICLES PER FRAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class Particle:
    def __init__(
        self, batch: Batch, position: np.ndarray, velocity: np.ndarray, radius: float
    ):
        self.alive = False
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.shape = pyglet.shapes.Circle(
            self.position[0],
            self.position[1],
            self.radius,
            color=(255, 255, 255),
            batch=batch,
        )

    def reset(self):
        self.alive = True
        self.position[:] = CENTER
        speed = np.random.uniform(200, 300)
        angle = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
        self.velocity[0] = speed * np.cos(angle)
        self.velocity[1] = speed * np.sin(angle)

    def update(self, dt, wind_velocity_x):
        if self.position[1] < 0:
            self.alive = False

        wind_accel = wind_velocity_x * 2 * self.radius - self.velocity[0]
        self.velocity[0] += wind_accel * dt
        self.velocity[1] -= GRAVITY_ACCELERATION * dt
        self.position += self.velocity * dt

        # update the position
        self.shape.x = self.position[0]
        self.shape.y = self.position[1]


def main() -> None:
    g_particles = []
    for _ in range(START_PARTICLES):
        g_particles.append(
            Particle(
                batch=g_batch,
                position=np.zeros(2),
                velocity=np.zeros(2),
                radius=np.random.uniform(10, 15),
            )
        )

    @g_window.event
    def on_draw():
        g_window.clear()
        g_batch.draw()

    def update(dt):
        time = 0
        time += dt
        wind_velocity_x = np.sin(time) * 70

        for particle in g_particles:
            if not particle.alive:
                particle.reset()

            particle.update(dt, wind_velocity_x)

    pyglet.clock.schedule_interval(update, FRAME_TIME)
    pyglet.app.run()


if __name__ == "__main__":
    main()
