import pyglet
from pyglet.math import Vec2
from pyglet.window import key


# testingmichelle

window_width = 600
window_height = 800
dt: float = 1 / 60

window = pyglet.window.Window(window_width, window_height)
batch = pyglet.graphics.Batch()
maingroup = pyglet.graphics.Group(order=1)
overgroup = pyglet.graphics.Group(order=0)

# Use later
group = pyglet.graphics.Group()

# To check the state of the current key (which is pressed)
keys = key.KeyStateHandler()
window.push_handlers(keys)

brown: tuple[int, int, int] = (153, 76, 0)

green: tuple[int, int, int] = (0, 153, 0)

pink: tuple[int, int, int] = (255, 153, 204)


class Rectangle:

    def __init__(self, width, height, color, batch, group=maingroup, pos=Vec2(0, 0)):

        self.pos = pos

        self.width = width

        self.height = height

        self.color = color

        self.shape = pyglet.shapes.Rectangle(
            self.pos.x,
            self.pos.y,
            self.width,
            self.height,
            self.color,
            batch=batch,
            group=group,
        )


# Class of State Cat
class Cat:
    body_parts: list[Rectangle]
    body_part_index: list[tuple[int, int, int]]
    main_body: Rectangle

    def __init__(
        self, body_parts: list[Rectangle], body_part_index: list[tuple[int, int, int]]
    ):
        self.body_parts = body_parts
        self.body_part_index = body_part_index
        # Checking if there is a main body
        self.main_body: Rectangle = self.body_parts[0]

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


cat_body_parts: list[Rectangle] = [
    Rectangle(
        pos=Vec2(200, 200),
        width=100,
        height=100,
        color=brown,
        batch=batch,
    ),  # main body
    Rectangle(width=50, height=50, color=pink, batch=batch),
    Rectangle(width=50, height=50, color=green, batch=batch),
    Rectangle(width=150, height=50, color=pink, batch=batch),
    Rectangle(
        width=50, height=50, color=pink, batch=batch, group=overgroup
    ),  # this one goes over
]


cat_body_index: list[tuple[int, int, int]] = [
    (0, 0, -1),
    (-50, -50, 0),
    (50, -50, 0),
    (-150, 50, 0),
    (10, 10, 0),
]
cat = Cat(cat_body_parts, cat_body_index)


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
