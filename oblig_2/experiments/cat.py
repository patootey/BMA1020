import pyglet
from pyglet.math import Vec2
from pyglet.window import key

import math

# NEWEST for the cat state

"""
batch?
clamp + cirlce / line collision
main body, gets drawn but not changed accordingly
    - index, is it right?
    - id with offset -1 just bounces everywhere
body_circles, smart måte? eller bedre måte å gjøre det på?
shoudl i have the start screen, maybe not (not in descripton)

set the velocity to be primarilyt 0? so i dont have to do it here
"""

# testing B task

window_width = 600
window_height = 800
dt: float = 1 / 60

window = pyglet.window.Window(window_width, window_height)
batch = pyglet.graphics.Batch()

# Use later
group = pyglet.graphics.Group()

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
        self,
        pos: Vec2 | None = None,
        velocity: Vec2 | None = None,
        radius: float = 1,
        color: tuple[int, int, int] = color1,
    ) -> None:
        self.pos = pos or Vec2()
        self.velocity = velocity or Vec2()
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


# Class of State Cat
class Cat:
    body_parts: list[Circle]
    body_part_index: list[tuple[int, int, int]]
    main_body: Circle

    def __init__(
        self, body_parts: list[Circle], body_part_index: list[tuple[int, int, int]]
    ):
        self.body_parts = body_parts
        self.body_part_index = body_part_index
        # Checking if there is a main body
        main_body: Circle | None = None
        for part_id, part in enumerate(self.body_part_index):
            if part[2] == -1:
                main_body = self.body_parts[part_id]  # main body
        if main_body is None:
            raise Exception("Could not find main body")
        self.main_body = main_body

    def update(self, dt: float):

        # copy
        # got help from my dad
        self.main_body.shape.position = self.main_body.pos

        # find each index
        for part_id, (rel_x, rel_y, parent_id) in enumerate(self.body_part_index):
            body_part = self.body_parts[part_id]
            if parent_id == -1:  # skip over main body part
                continue
            while parent_id != -1:
                parent_rel_x, parent_rel_y, parent_id = self.body_part_index[parent_id]
                if parent_id != -1:
                    rel_x += parent_rel_x
                    rel_y += parent_rel_y

            body_part.pos = Vec2(
                self.main_body.pos.x + rel_x, self.main_body.pos.y + rel_y
            )
            body_part.shape.position = (body_part.pos.x, body_part.pos.y)


# State holder
states: tuple[str, str, str] = ("Start_state", "A_state", "B_state")
current_state = states[2]

cat_body_parts: list[Circle] = [
    Circle(
        pos=Vec2(window.width / 2, window.height / 2), radius=100, color=color1
    ),  # main body
    Circle(radius=30, color=color2),  # head
    Circle(radius=10, color=color3),  # leg1
    Circle(radius=10, color=color3),  # leg2
    Circle(radius=5, color=color1),  # toe
]


cat_body_part_index: list[tuple[int, int, int]] = [
    (0, 0, -1),  # Main body (no offset)
    (0, 110, 0),  # Head
    (-30, -100, 0),  # Leg_left
    (30, -100, 0),  # Leg_right
    (-40, -110, 2),  # toe
]

cat = Cat(cat_body_parts, cat_body_part_index)


@window.event
def on_draw():
    window.clear()

    if current_state == "B_state":
        batch.draw()


def update(dt: float):
    global current_state

    if current_state == "B_state":
        if keys[key.DOWN]:
            cat.body_parts[0].pos = Vec2(
                cat.body_parts[0].pos.x, cat.body_parts[0].pos.y - 200 * dt
            )
        if keys[key.UP]:
            cat.body_parts[0].pos = Vec2(
                cat.body_parts[0].pos.x, cat.body_parts[0].pos.y + 200 * dt
            )
        if keys[key.LEFT]:
            cat.body_parts[0].pos = Vec2(
                cat.body_parts[0].pos.x - 200 * dt, cat.body_parts[0].pos.y
            )
        if keys[key.RIGHT]:
            cat.body_parts[0].pos = Vec2(
                cat.body_parts[0].pos.x + 200 * dt, cat.body_parts[0].pos.y
            )

        cat.update(dt)


# Schedule updates
pyglet.clock.schedule_interval(update, dt)

# Run the app
pyglet.app.run()
