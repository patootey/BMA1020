import pyglet
from pyglet.math import Vec2
from pyglet.window import key
import random
import math

# NEWEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

window_width = 600
window_height = 800
dt: float = 1 / 60

window = pyglet.window.Window(window_width, window_height)
batch = pyglet.graphics.Batch()
# Use later
# group = pyglet.graphics.Group()

# To check the state of the current key (which is pressed)
keys = key.KeyStateHandler()
window.push_handlers(keys)

color1: tuple[int, int, int] = (123, 66, 151)  # purple
color2: tuple[int, int, int] = (232, 138, 89)  # orange
color3: tuple[int, int, int] = (24, 172, 226)  # blue
color4: tuple[int, int, int] = (226, 24, 85)  # pink


# Circle CLass
class Circle:
    pos: Vec2
    radius: float
    velocity: Vec2
    color: tuple[int, int, int]

    def __init__(
        self, pos: Vec2, velocity: Vec2, radius: float, color: tuple[int, int, int]
    ) -> None:
        self.pos = pos
        self.velocity = velocity
        self.radius = radius
        self.color = color
        self.shape = pyglet.shapes.Circle(
            self.pos.x,
            self.pos.y,
            self.radius,
            color=self.color,
            batch=batch,
        )

    def update(self, dt: float):
        self.shape.x += dt * self.velocity.x
        self.shape.y += dt * self.velocity.y

        if self.shape.x - self.shape.radius > window.width:
            self.shape.x = -self.shape.radius
        elif self.shape.x + self.shape.radius < 0:
            self.shape.x = window.width + self.shape.radius

        if self.shape.y - self.shape.radius > window.height:
            self.shape.y = -self.shape.radius
        elif self.shape.y + self.shape.radius < 0:
            self.shape.y = window.height + self.shape.radius

    def check_collision(self, other: "Circle") -> bool:
        # Calculate the distance between the two circles
        # Got help with the mathematic formula from my dad, showed me where to look
        distance: float = math.sqrt(
            ((self.shape.x - other.shape.x) ** 2 + (self.shape.y - other.shape.y) ** 2)
        )
        return distance < (self.radius + other.radius)

    def set_color(self, color: tuple[int, int, int]):
        self.shape.color = color


# Line class
class Line:
    # endre
    def __init__(
        self,
        pos_1: Vec2,
        pos_2: Vec2,
        velocity: Vec2,
        thickness: float,
        color: tuple[int, int, int],
    ) -> None:
        self.pos_1 = pos_1
        self.pos_2 = pos_2
        self.velocity = velocity
        self.thickness = thickness
        self.color: tuple[int, int, int] = color

        self.shape = pyglet.shapes.Line(
            self.pos_1.x,
            self.pos_1.y,
            self.pos_2.x,
            self.pos_2.y,
            self.thickness,
            color=self.color,
            batch=batch,
        )

    def update(self, dt: float):
        self.shape.x += dt * self.velocity.x
        self.shape.x2 += dt * self.velocity.x
        self.shape.y += dt * self.velocity.y
        self.shape.y2 += dt * self.velocity.y

        # Wrap x-coordinates
        if self.shape.x > window.width and self.shape.x2 > window.width:
            self.shape.x = -window.width
            self.shape.x2 -= window.width
        elif self.shape.x < 0 and self.shape.x2 < 0:
            self.shape.x += window.width
            self.shape.x2 += window.width

        # Wrap y-coordinates
        if self.shape.y > window.height and self.shape.y2 > window.height:
            self.shape.y -= window.height
            self.shape.y2 -= window.height
        elif self.shape.y < 0 and self.shape.y < 0:
            self.shape.y += window.height
            self.shape.y2 += window.height

    def check_collision(self, other: "Line"):
        # add more
        # vec_1 = Vec2(self.shape.x, self.shape.y)
        # vec_2 = Vec2(other.shape.x, other.shape.y)
        # vec_4 = Vec2(self.shape.x2, self.shape.y2)
        # vec_5 = Vec2(other.shape.x2, other.shape.y2)
        # if vec_1.distance(vec_2) and vec_4.distance(vec_5):
        #

        pass

    def set_color(self, color: tuple[int, int, int]):
        self.shape.color = color


# Class of State A (which controls Circles and Lines)
class A:
    def __init__(self, circles: list[Circle], lines: list[Line]):
        self.circles: list[Circle] = circles
        self.lines: list[Line] = lines
        self.label = pyglet.text.Label(
            "Press Enter To Start",
            font_name="Times New Roman",
            font_size=36,
            x=window.width // 2,
            y=window.height // 2,
            anchor_x="center",
            anchor_y="center",
        )

    def start_screen(self):
        self.label.draw()

    def update(self, dt: float):
        # Update circles and lines
        for circle in self.circles:
            circle.update(dt)

        for line in self.lines:
            line.update(dt)

        for circle in self.circles:
            circle.set_color(color1)

        # Check collisions between circles
        for i in range(len(self.circles)):
            for j in range(i + 1, len(self.circles)):
                if self.circles[i].check_collision(self.circles[j]):
                    self.circles[i].set_color(color3)
                    self.circles[j].set_color(color3)

        # Check collisions between lines
        for i in range(len(self.lines)):
            for j in range(i + 1, len(self.lines)):
                if self.lines[i].check_collision(self.lines[j]):
                    self.lines[i].set_color(color4)
                    self.lines[j].set_color(color4)

        # check between circles and lines
        # use vector?


# Class of State Cat
# maybe a subclass?
# group (Group | None) – Optional parent group of the shape. (for the parent)
class Cat:
    def __init__(self):
        self.body: list = []

    def update(self, dt):
        # the parent
        pass

    def draw(self):
        # group draw() ??
        pass


# State holder
states: tuple[str, str] = ("Start_state", "A_state")
current_state = states[0]


circles: list[Circle] = []
lines: list[Line] = []

for i in range(3):
    lines.append(
        Line(
            pos_1=(
                Vec2(random.uniform(0, window.width), random.uniform(0, window.height))
            ),
            pos_2=(
                Vec2(random.uniform(0, window.width), random.uniform(0, window.height))
            ),
            velocity=(Vec2(100, 100)),
            thickness=20,
            color=color2,
        )
    )

for i in range(5):
    circles.append(
        Circle(
            pos=Vec2(random.uniform(0, window.width), random.uniform(0, window.height)),
            velocity=Vec2(random.uniform(10, 500), random.uniform(50, 500)),
            radius=random.uniform(10, 50),
            color=color1,
        )
    )

a_state = A(circles, lines)


@window.event
def on_draw():
    window.clear()

    if current_state == "Start_state":
        a_state.start_screen()

    if current_state == "A_state":
        batch.draw()


def update(dt: float):
    global current_state

    if keys[key.ENTER]:
        current_state = states[1]

    # just to test, for future use
    if keys[key.T]:
        current_state = states[0]

    if current_state == "A_state":
        a_state.update(dt)


# Schedule updates
pyglet.clock.schedule_interval(update, dt)

# Run the app
pyglet.app.run()
