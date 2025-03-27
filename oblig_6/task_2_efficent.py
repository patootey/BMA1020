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


g_batch = Batch()
g_window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

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
        self.time = 0

    def reset(self):
        dead_indices = np.where(~self.alive)[0][:PARTICLES_PER_FRAME]
        needed = PARTICLES_PER_FRAME - len(dead_indices)
        if needed > 0:
            new_dead = np.arange(len(self.alive), len(self.alive) + needed)
            dead_indices = np.append(dead_indices, new_dead)

            self.alive = np.append(self.alive, np.zeros(needed, dtype=bool))
            self.positions = np.append(self.positions, np.zeros((needed, 2)), axis=0)
            self.velocities = np.append(self.velocities, np.zeros((needed, 2)), axis=0)
            radius = np.random.uniform(10, 15, needed)
            self.shapes = np.append(
                self.shapes,
                np.array(
                    [
                        shapes.Circle(
                            0, 0, radius[i], color=(255, 255, 255), batch=g_batch
                        )
                        for i in range(needed)
                    ]
                ),
            )
            self.radiuses = np.append(self.radiuses, radius)
        self.alive[dead_indices] = True
        shape: shapes.Circle
        for shape in self.shapes[dead_indices]:
            shape.visible = True
        self.positions[dead_indices, :] = CENTER
        speed = np.random.uniform(200, 300, len(dead_indices))
        angle = np.random.uniform(np.pi / 4, 3 * np.pi / 4, len(dead_indices))
        self.velocities[dead_indices, 0] = speed * np.cos(angle)
        self.velocities[dead_indices, 1] = speed * np.sin(angle)
        # for i in dead_indices:
        #     self.alive[i] = True
        #     self.positions[i, :] = CENTER
        #     speed = np.random.uniform(200, 300)
        #     angle = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
        #     self.velocities[i, 0] = speed * np.cos(angle)
        #     self.velocities[i, 1] = speed * np.sin(angle)

    def update(self, dt: float):
        self.time += dt
        make_dead = np.where(self.positions[:, 1] < 0)
        self.alive[make_dead] = False
        shape: shapes.Circle
        for shape in self.shapes[make_dead]:
            shape.visible = False

        are_alive = np.where(self.alive)
        wind_velocity_x = np.sin(self.time) * 20
        wind_accel = (
            wind_velocity_x * 2 * self.radiuses[are_alive]
            - self.velocities[are_alive, 0]
        )
        self.velocities[are_alive, 0] += wind_accel * dt
        self.velocities[are_alive, 1] -= GRAVITY_ACCELERATION * dt
        # for i in are_alive:
        #     wind_velocity_x = np.sin(self.time) * 20
        #     wind_accel = wind_velocity_x * 2 * self.radiuses[i] - self.velocities[i, 0]
        #     self.velocities[i, 0] += wind_accel * dt
        #     self.velocities[i, 1] -= GRAVITY_ACCELERATION * dt

        self.positions[are_alive] += self.velocities[are_alive] * dt

        for i, shape in enumerate(self.shapes):
            shape.x = self.positions[i, 0]
            shape.y = self.positions[i, 1]


def main() -> None:
    alive: np.ndarray = np.zeros(START_PARTICLES, dtype=bool)
    velocity: np.ndarray = np.zeros((START_PARTICLES, 2))
    position: np.ndarray = np.zeros((START_PARTICLES, 2))
    radius: np.ndarray = np.random.uniform(10, 15, START_PARTICLES)
    circle_shapes: np.ndarray = np.array(
        [
            shapes.Circle(0, 0, radius[i], color=(255, 255, 255), batch=g_batch)
            for i in range(START_PARTICLES)
        ]
    )

    label = pyglet.text.Label(
        "",
        font_name="Times New Roman",
        font_size=36,
        x=100,
        y=100,
        anchor_x="center",
        anchor_y="center",
    )
    particles = ParticleSystem(
        batch=g_batch,
        alive=alive,
        shapes=circle_shapes,
        position=position,
        radius=radius,
        velocity=velocity,
    )

    @g_window.event
    def on_draw():
        g_window.clear()
        g_batch.draw()
        label.draw()

    def update(dt):
        particles.reset()
        particles.update(dt)
        label.text = f"{len(particles.alive)}"

    pyglet.clock.schedule_interval(update, FRAME_TIME)
    pyglet.app.run()


if __name__ == "__main__":
    main()
