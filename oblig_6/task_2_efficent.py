import pyglet
from pyglet.graphics import Batch
from pyglet import shapes
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

g_batch = Batch()
g_window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

g_alive: np.ndarray = np.zeros(START_PARTICLES, dtype=bool)
g_velocity: np.ndarray = np.zeros((START_PARTICLES, 2))
g_position: np.ndarray = np.zeros((START_PARTICLES, 2))
g_radius: np.ndarray = np.random.uniform(10, 15, START_PARTICLES)
g_shapes: np.ndarray = np.array(
    [
        shapes.Circle(0, 0, g_radius[i], color=(255, 255, 255), batch=g_batch)
        for i in range(START_PARTICLES)
    ]
)


# gjør til funksjoner, sende inn particel system , med index
# WHY NO WOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOORK


# MANGLER 5 PARTICLES PER FRAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class ParticleSystem:
    def __init__(
        self,
        batch: Batch,
        alive: np.ndarray,
        shapes: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: np.ndarray,
    ):
        self.alive = alive
        self.positions = position
        self.velocities = velocity
        self.radiuses = radius
        self.shapes = shapes
        self.batch = batch

    def reset(self):
        not_alive_indices = np.where(~self.alive)
        for i in not_alive_indices:
            self.alive[i] = True
            self.positions[i, :] = CENTER
            speed = np.random.uniform(200, 300)
            angle = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
            self.velocities[i, 0] = speed * np.cos(angle)
            self.velocities[i, 1] = speed * np.sin(angle)

    def update(self, dt: float):
        time = 0
        time += dt
        for i in range(START_PARTICLES):
            if self.positions[i, 1] < 0:
                self.alive[i] = False
                continue

            wind_velocity_x = np.sin(time) * 70
            wind_accel = wind_velocity_x * 2 * self.radiuses[i] - self.velocities[i, 0]
            self.velocities[i, :] += wind_accel * dt
            self.velocities[i, :] -= GRAVITY_ACCELERATION * dt
            self.positions += self.velocities * dt


def main() -> None:
    particles = ParticleSystem(
        batch=g_batch,
        alive=g_alive,
        shapes=g_shapes,
        position=g_position,
        radius=g_radius,
        velocity=g_velocity,
    )

    @g_window.event
    def on_draw():
        g_window.clear()
        for i, shape in enumerate(particles.shapes):
            shape.x = particles.positions[i, 0]
            shape.y = particles.positions[i, 1]
        g_batch.draw()

    def update(dt):
        particles.reset()
        particles.update(dt)

    pyglet.clock.schedule_interval(update, FRAME_TIME)
    pyglet.app.run()


if __name__ == "__main__":
    main()
