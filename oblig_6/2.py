import pyglet
import numpy as np

# Constants
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
# START GRENSE
# allokerer på forhånd
START_PARTICLES = 100
GRAVITY_ACCELERATION = 200
CENTER = np.array([WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2])

# alltid legge til 5 per sekund

# bruk istedenfor
g_batch = pyglet.graphics.Batch()


# numpy array, om de er i live
# numpy velocity
# numpy posisjon
# array med shapes
g_alive = np.array([])
g_velocity = np.array([])
g_position = np.array([])
g_shapes = np.array([])
g_radius = np.array([])


# windvel function?


def wind_accel(radius, velocity_x):
    return


# gjør til funksjoner, sende inn particel system , med index


# bytt ut
class Particle:
    def __init__(self, batch):
        self.shape = pyglet.shapes.Circle(0, 0, 0, color=(255, 255, 255), batch=batch)
        self.alive = False
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.radius = 0

    def reset(self):
        self.alive = True
        self.position[:] = CENTER
        speed = np.random.uniform(200, 300)
        angle = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
        self.velocity[0] = speed * np.cos(angle)
        self.velocity[1] = speed * np.sin(angle)
        self.radius = np.random.uniform(10, 15)
        self.shape.radius = self.radius

    def update(self, dt, wind_velocity_x):
        if not self.alive:
            return

        if self.position[1] < 0:
            self.alive = False
            return

        wind_accel = wind_velocity_x * 2 * self.radius - self.velocity[0]
        self.velocity[0] += wind_accel * dt
        self.velocity[1] -= GRAVITY_ACCELERATION * dt
        self.position += self.velocity * dt

        self.shape.x = self.position[0]
        self.shape.y = self.position[1]


class ParticleSystem:
    def __init__(self):
        self.batch = pyglet.graphics.Batch()
        self.particles = []

        # numpy where (finne index for boolean)
        # array med radius
        # update for MANGE
        # np.copyto() where=new_dead
        for _ in range(START_PARTICLES):
            self.particles.append(Particle(self.batch))
        self.time = 0.0

    def update(self, dt):
        self.time += dt
        wind_velocity_x = np.sin(self.time) * 100

        for particle in self.particles:
            if not particle.alive:
                particle.reset()

            particle.update(dt, wind_velocity_x)

    def draw(self):
        self.batch.draw()


def main():
    window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
    particle_system = ParticleSystem()

    @window.event
    def on_draw():
        window.clear()
        # particle_system.draw()
        g_batch.draw()

    def update(dt):
        particle_system.update(dt)

    pyglet.clock.schedule_interval(update, 1 / 60.0)
    pyglet.app.run()


if __name__ == "__main__":
    main()
