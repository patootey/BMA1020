import pyglet
from pyglet.math import Vec2, clamp
from pyglet.graphics import Batch, Group
from pyglet.window import key, Window
import random
import math

window_width = 600
window_height = 800
frame_time: float = 1 / 60

window = Window(window_width, window_height)
batch_a = Batch()
batch_b = Batch()
# Order of the drawing
last_group = Group(order=0)
normal_group = Group(order=1)

# To check the state of the current key (which is pressed)
keys = key.KeyStateHandler()
window.push_handlers(keys)

purple: tuple[int, int, int] = (123, 66, 151)  # purple
orange: tuple[int, int, int] = (232, 138, 89)  # orange
blue: tuple[int, int, int] = (24, 172, 226)  # blue
pink: tuple[int, int, int] = (226, 24, 85)  # pink
black: tuple[int, int, int] = (0, 0, 0)  # black


# Circle class: defines a circle with a center point and a radius
class Circle:
    pos: Vec2
    radius: float
    velocity: Vec2
    color: tuple[int, int, int]

    def __init__(
        self,
        batch: Batch,
        group: Group,
        color: tuple[int, int, int],
        pos: Vec2 | None = None,
        velocity: Vec2 | None = None,
        radius: float = 1,
    ) -> None:
        self.pos = pos or Vec2()
        self.velocity = velocity or Vec2()
        self.radius = radius
        self.color = color
        # The circle object drawn on the screen
        self.shape = pyglet.shapes.Circle(
            self.pos.x,
            self.pos.y,
            self.radius,
            color=self.color,
            batch=batch,
            group=group,
        )

    def update(self, dt: float):
        # Checks if circles should wrap around screen
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

    def set_color(self, color: tuple[int, int, int]):
        self.shape.color = color


# Line class: defines a line with different coordinates and a thickness
class Line:
    pos_1: Vec2
    pos_2: Vec2
    velocity: Vec2
    color: tuple[int, int, int]
    thickness: float

    def __init__(
        self,
        batch: Batch,
        group: Group,
        pos_1: Vec2 | None = None,
        pos_2: Vec2 | None = None,
        velocity: Vec2 | None = None,
        thickness: float = 1,
        color: tuple[int, int, int] = orange,
    ) -> None:
        self.pos_1 = pos_1 or Vec2()
        self.pos_2 = pos_2 or Vec2()
        self.velocity = velocity or Vec2()
        self.thickness = thickness
        self.color = color
        # The line object drawn on the screen
        self.shape = pyglet.shapes.Line(
            self.pos_1.x,
            self.pos_1.y,
            self.pos_2.x,
            self.pos_2.y,
            self.thickness,
            color=self.color,
            batch=batch,
            group=group,
        )

    def update(self, dt: float):
        # Checks if lines should wrap around screen
        x, y = self.shape.position
        x2, y2 = self.shape.x2, self.shape.y2

        # Used my dad + Curry lecture to find the "length"
        width = abs(x - x2)
        height = abs(y - y2)

        rel_x = self.velocity.x * dt
        rel_y = self.velocity.y * dt
        x += rel_x
        y += rel_y
        x2 += rel_x
        y2 += rel_y

        # Wrap x-coordinates
        if min(x, x2) > window.width and rel_x > 0:
            x -= window.width + width
            x2 -= window.width + width

        elif max(x, x2) < 0 and rel_x < 0:
            x += window.width + width
            x2 += window.width + width

        # Wrap y-coordinates
        if min(y, y2) > window.height and rel_y > 0:
            y -= window.height + height
            y2 -= window.height + height

        elif max(y, y2) < 0 and rel_y < 0:
            y += window.height + height
            y2 += window.height + height

        # Write back to shape
        self.shape.position = (x, y)
        self.shape.x2 = x2
        self.shape.y2 = y2

    def set_color(self, color: tuple[int, int, int]):
        self.shape.color = color


# Rectangle class: defines a rectangel with coordinates, width and height
class Rectangle:
    pos: Vec2
    width: float
    height: float
    color: tuple[int, int, int]

    def __init__(
        self,
        width: float,
        height: float,
        color: tuple[int, int, int],
        batch: Batch,
        group: Group,
        pos: Vec2 | None = None,
    ):
        self.pos = pos or Vec2()
        self.width = width
        self.height = height
        self.color = color
        # The rectangle object drawn on screen
        self.shape = pyglet.shapes.Rectangle(
            self.pos.x,
            self.pos.y,
            self.width,
            self.height,
            self.color,
            batch=batch,
            group=group,
        )


# Class WrapAroundScreen (which controls Circles and Lines)
class WrapAroundScreen:
    circles: list[Circle]
    lines: list[Line]

    def __init__(self, circles: list[Circle], lines: list[Line]):
        self.circles = circles
        self.lines = lines
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
            circle.set_color(purple)

        # Check collisions between circles and lines / circle - cirle
        # Using bool
        for circle in self.circles:
            collides_with_line = False
            collides_with_circle = False

            # Check if the circle collides with any line
            for line in self.lines:
                if check_collision_circle_line(circle, line):
                    collides_with_line = True
                    break

            # Check if the circle collides with any other circle
            for other_circle in self.circles:
                if other_circle is not circle and check_collision_circle_circle(
                    circle, other_circle
                ):
                    collides_with_circle = True
                    break

            # Sets the correct colors
            if collides_with_line and collides_with_circle:
                circle.set_color(orange)
                other_circle.set_color(orange)
            elif collides_with_line:
                circle.set_color(pink)
            elif collides_with_circle:
                circle.set_color(blue)


# Class of State Cat
class Cat:
    body_parts: list[Circle | Rectangle]
    body_part_index: list[tuple[int, int, int]]
    main_body: Circle | Rectangle

    def __init__(
        self,
        body_parts: list[Circle | Rectangle],
        body_part_index: list[tuple[int, int, int]],
    ):
        self.body_parts = body_parts
        self.body_part_index = body_part_index
        # Checking if there is a main body
        main_body: Circle | Rectangle | None = None
        # Just MAKES SURE that there is a mainbody (or else the code wont run)
        for part_id, part in enumerate(self.body_part_index):
            if part[2] == -1:
                main_body = self.body_parts[part_id]  # main body
                break
        if main_body is None:
            raise Exception("Could not find main body")
        self.main_body = main_body

    def update(self, dt: float):
        self.main_body.shape.position = self.main_body.pos

        # Find each index
        for part_id, (rel_x, rel_y, parent_id) in enumerate(self.body_part_index):
            body_part = self.body_parts[part_id]
            # Got help from my dad for the logic (parent-parent)
            if parent_id == -1:  # Skip over main body part
                continue
            while parent_id != -1:
                # Add the parent_relative x and y to the each elemements relative postition
                parent_rel_x, parent_rel_y, parent_id = self.body_part_index[parent_id]
                if parent_id != -1:
                    rel_x += parent_rel_x
                    rel_y += parent_rel_y

            # Send the new coordinates to the circle correct object
            body_part.pos = Vec2(
                self.main_body.pos.x + rel_x, self.main_body.pos.y + rel_y
            )
            body_part.shape.position = (body_part.pos.x, body_part.pos.y)


# Collision checks functions
def check_collision_circle_circle(circle_a: "Circle", circle_b: "Circle") -> bool:
    # Calculate the distance between the two circles
    # Found the formula on the net
    distance: float = math.sqrt(
        (
            (circle_a.shape.x - circle_b.shape.x) ** 2
            + (circle_a.shape.y - circle_b.shape.y) ** 2
        )
    )
    return distance < (circle_a.radius + circle_b.radius)


def check_collision_circle_line(circle_a: Circle, line_b: Line) -> bool:
    # Some logic help from my dad (mostly the lecture from Curry)
    circle_center = Vec2(circle_a.shape.x, circle_a.shape.y)
    line_start = Vec2(line_b.shape.x, line_b.shape.y)
    line_end = Vec2(line_b.shape.x2, line_b.shape.y2)

    line_vec = line_end - line_start
    circle_vec = circle_center - line_start
    # Using Vec2 built in methods
    line_length = line_vec.length()

    if line_length == 0:
        return circle_vec.length() <= circle_a.radius

    # Project the circle vector onto the line vector using clamped t
    t = clamp(circle_vec.dot(line_vec) / line_length**2, 0, 1)
    projection = line_start + t * line_vec

    # Return a bool if it is less than or equal the radius
    return (circle_center - projection).length() <= circle_a.radius


# State holder
states: tuple[str, str, str] = ("Start_state", "A_state", "B_state")
current_state = "Start_state"

# Circles and lines
circles: list[Circle] = []
lines: list[Line] = []

# Set up 4 lines with random positions, velocities, orientations and sizes
for i in range(4):
    lines.append(
        Line(
            pos_1=(
                Vec2(random.uniform(0, window.width), random.uniform(0, window.height))
            ),
            pos_2=(
                Vec2(random.uniform(0, window.width), random.uniform(0, window.height))
            ),
            velocity=Vec2(random.uniform(10, 400), random.uniform(50, 500)),
            thickness=random.uniform(5, 20),
            color=orange,
            batch=batch_a,
            group=normal_group,
        )
    )

# Set up 5 Circles with random positions, velocities and sizes
for i in range(5):
    circles.append(
        Circle(
            pos=Vec2(random.uniform(0, window.width), random.uniform(0, window.height)),
            velocity=Vec2(random.uniform(10, 400), random.uniform(50, 500)),
            radius=random.uniform(10, 50),
            color=purple,
            batch=batch_a,
            group=normal_group,
        )
    )

# Cat body part and index
cat_body_parts: list[Circle | Rectangle] = [
    Circle(
        pos=Vec2(window.width / 2, window.height / 2),
        radius=100,
        color=purple,
        batch=batch_b,
        group=normal_group,
    ),  # main body
    Circle(radius=50, color=orange, batch=batch_b, group=normal_group),  # head
    Circle(radius=10, color=blue, batch=batch_b, group=last_group),  # leg1
    Circle(radius=10, color=blue, batch=batch_b, group=normal_group),  # leg2
    Circle(radius=10, color=blue, batch=batch_b, group=last_group),  # leg3
    Circle(radius=10, color=blue, batch=batch_b, group=normal_group),  # leg4
    Circle(radius=10, color=blue, batch=batch_b, group=last_group),  # thigh1
    Circle(radius=10, color=blue, batch=batch_b, group=normal_group),  # thigh2
    Circle(radius=10, color=blue, batch=batch_b, group=normal_group),  # thigh3
    Circle(radius=10, color=blue, batch=batch_b, group=normal_group),  # thigh4
    Rectangle(
        width=10, height=10, color=pink, batch=batch_b, group=normal_group
    ),  # paw1
    Rectangle(
        width=10, height=10, color=pink, batch=batch_b, group=normal_group
    ),  # paw2
    Rectangle(
        width=10, height=10, color=pink, batch=batch_b, group=normal_group
    ),  # paw3
    Rectangle(
        width=10, height=10, color=pink, batch=batch_b, group=normal_group
    ),  # paw4
    Rectangle(
        width=30, height=30, color=purple, batch=batch_b, group=normal_group
    ),  # ear1
    Rectangle(
        width=30, height=30, color=purple, batch=batch_b, group=normal_group
    ),  # ear2
    Circle(radius=20, color=blue, batch=batch_b, group=last_group),  # tail1
    Circle(radius=20, color=blue, batch=batch_b, group=normal_group),  # tail2
    Rectangle(
        width=30, height=30, color=pink, batch=batch_b, group=normal_group
    ),  # tail 3
    Circle(radius=10, color=black, batch=batch_b, group=last_group),  # eye_left
    Circle(radius=10, color=black, batch=batch_b, group=last_group),  # eye_right
    Circle(radius=5, color=pink, batch=batch_b, group=last_group),  # nose
]


cat_body_part_index: list[tuple[int, int, int]] = [
    (0, 0, -1),  # Main body    (0)
    (-50, 120, 0),  # Head      (1)
    (-55, -80, 0),  # Leg1      (2)
    (-30, -95, 0),  # Leg2      (3)
    (30, -95, 0),  # Leg3       (4)
    (60, -75, 0),  # Leg4       (5)
    (-10, -15, 2),  # thigh1    (6)
    (-7, -17, 3),  # thigh2     (7)
    (10, -17, 4),  # thigh3     (8)
    (10, -15, 5),  # thigh4     (9)
    (-12, -12, 6),  # paw1      (10)
    (-10, -14, 7),  # paw2      (11)
    (2, -14, 8),  # paw3        (12)
    (2, -14, 9),  # paw4        (13)
    (-45, 15, 1),  # ear1       (14)
    (15, 15, 1),  # ear2        (15)
    (105, 30, 0),  # tail1      (16)
    (30, 20, 16),  # tail2      (17)
    (0, -5, 17),  # tail3       (18)
    (-20, 0, 1),  # eye_left    (19)
    (20, 0, 1),  # eye_left     (20)
    (0, -10, 1),  # nose        (21)
]

# Defining the cat/wraparoundscreen objects
cat = Cat(cat_body_parts, cat_body_part_index)
wrap = WrapAroundScreen(circles, lines)


@window.event
def on_draw():
    window.clear()

    # Draw depending on each state
    if current_state == "Start_state":
        wrap.start_screen()

    elif current_state == "A_state":
        batch_a.draw()

    elif current_state == "B_state":
        batch_b.draw()


def update(dt: float):
    global current_state

    # Updating for the screen
    if current_state == "Start_state":
        pass

    elif current_state == "A_state":
        wrap.update(dt)

    elif current_state == "B_state":
        # Movement for the cat using vector postitions
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


def on_key_release(symbol, modifiers):
    global current_state

    # State switching
    if current_state == "Start_state":
        if symbol == key.ENTER:
            current_state = "A_state"

    elif current_state == "A_state":
        if symbol == key.SPACE:
            current_state = "B_state"

    elif current_state == "B_state":
        if symbol == key.SPACE:
            current_state = "A_state"


# For the keys that is to be pressed once
window.push_handlers(on_key_release)
# Schedule updates
pyglet.clock.schedule_interval(update, frame_time)
# Run the app
pyglet.app.run()
