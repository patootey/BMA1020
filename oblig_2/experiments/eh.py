import pyglet
from pyglet.math import Vec2, clamp
from pyglet.graphics import Batch, Group
from pyglet.window import key
import random
import math

# testing 2

window_width = 600
window_height = 800
dt: float = 1 / 60

window = pyglet.window.Window(window_width, window_height)
batch_a = Batch()
batch_b = Batch()
cat_face_group = Group(order=0)
normal_group = Group(order=1)
# To check the state of the current key (which is pressed)
keys = key.KeyStateHandler()
window.push_handlers(keys)

color1: tuple[int, int, int] = (123, 66, 151)  # purple
color2: tuple[int, int, int] = (232, 138, 89)  # orange
color3: tuple[int, int, int] = (24, 172, 226)  # blue
color4: tuple[int, int, int] = (226, 24, 85)  # pink
black: tuple[int, int, int] = (0, 0, 0)


# Circle CLass
class Circle:
    pos: Vec2
    radius: float
    velocity: Vec2
    color: tuple[int, int, int]

    def __init__(
        self,
        batch: Batch,
        group: Group,
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
            group=group,
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

    def set_color(self, color: tuple[int, int, int]):
        self.shape.color = color


# Line class
class Line:
    pos_1: Vec2
    pos_2: Vec2
    velocity: Vec2
    color: tuple[int, int, int]
    thickness: float

    def __init__(
        self,
        batch: Batch,
        pos_1: Vec2 | None = None,
        pos_2: Vec2 | None = None,
        velocity: Vec2 | None = None,
        thickness: float = 1,
        color: tuple[int, int, int] = color2,
    ) -> None:
        self.pos_1 = pos_1 or Vec2()
        self.pos_2 = pos_2 or Vec2()
        self.velocity = velocity or Vec2()
        self.thickness = thickness
        self.color = color

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
        # bool check
        x, y = self.shape.position
        x2, y2 = self.shape.x2, self.shape.y2

        width = abs(x - x2)
        height = abs(y - y2)

        rel_x = self.velocity.x * dt
        rel_y = self.velocity.y * dt
        x += rel_x
        y += rel_y
        x2 += rel_x
        y2 += rel_y

        # print(x, x2, min(x, x2), window.width)
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
            circle.set_color(color1)

        # Check collisions between circles
        for i in range(len(self.circles)):
            for j in range(i + 1, len(self.circles)):
                if check_collision_circle_circle(self.circles[i], self.circles[j]):
                    self.circles[i].set_color(color3)
                    self.circles[j].set_color(color3)

        # some logic help from my dad
        # Check collisions between circles and lines
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
                if other_circle != circle and check_collision_circle_circle(
                    circle, other_circle
                ):
                    collides_with_circle = True
                    break

            # If the circle collides with both a line and another circle, set its color to color2
            if collides_with_line and collides_with_circle:
                circle.set_color(color2)
            elif collides_with_line:
                circle.set_color(color4)


# Class of State Cat
class Cat:
    body_parts: list[Circle]
    body_part_index: list[tuple[int, int, int]]
    main_body: Circle

    def __init__(
        self,
        body_parts: list[Circle],
        body_part_index: list[tuple[int, int, int]],
    ):
        self.body_parts = body_parts
        self.body_part_index = body_part_index
        # Checking if there is a main body
        main_body: Circle | None = None
        for part_id, part in enumerate(self.body_part_index):
            if part[2] == -1:
                main_body = self.body_parts[part_id]  # main body
                break
        if main_body is None:
            raise Exception("Could not find main body")
        self.main_body = main_body

    def update(self, dt: float):

        # copy
        self.main_body.shape.position = self.main_body.pos

        # find each index
        for part_id, (rel_x, rel_y, parent_id) in enumerate(self.body_part_index):
            body_part = self.body_parts[part_id]
            # got help from my dad (parent-parent)
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


# global functions
def check_collision_circle_circle(circle_a: "Circle", circle_b: "Circle") -> bool:
    # Calculate the distance between the two circles
    # Got help with the mathematic formula from my dad, showed me where to look
    distance: float = math.sqrt(
        (
            (circle_a.shape.x - circle_b.shape.x) ** 2
            + (circle_a.shape.y - circle_b.shape.y) ** 2
        )
    )
    return distance < (circle_a.radius + circle_b.radius)


def check_collision_circle_line(circle_a: Circle, line_b: Line) -> bool:
    # some logic help from my dad (mostly the lecture from currry)
    circle_center = Vec2(circle_a.shape.x, circle_a.shape.y)  # Circle position
    line_start = Vec2(line_b.shape.x, line_b.shape.y)  # Start point of the line
    line_end = Vec2(line_b.shape.x2, line_b.shape.y2)  # End point of the line

    line_vec = line_end - line_start
    circle_vec = circle_center - line_start
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
current_state = "B_state"


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
            velocity=Vec2(random.uniform(10, 500), random.uniform(50, 500)),
            thickness=random.uniform(5, 20),
            color=color2,
            batch=batch_a,
        )
    )

for i in range(5):
    circles.append(
        Circle(
            pos=Vec2(random.uniform(0, window.width), random.uniform(0, window.height)),
            velocity=Vec2(random.uniform(10, 500), random.uniform(50, 500)),
            radius=random.uniform(10, 50),
            color=color1,
            batch=batch_a,
            group=normal_group,
        )
    )


cat_body_parts: list[Circle] = [
    Circle(
        pos=Vec2(window.width / 2, window.height / 2),
        radius=100,
        color=color1,
        batch=batch_b,
        group=normal_group,
    ),  # main body
    Circle(radius=50, color=color2, batch=batch_b, group=normal_group),  # head
    Circle(radius=10, color=color3, batch=batch_b, group=cat_face_group),  # leg1
    Circle(radius=10, color=color3, batch=batch_b, group=normal_group),  # leg2
    Circle(radius=10, color=color3, batch=batch_b, group=cat_face_group),  # leg3
    Circle(radius=10, color=color3, batch=batch_b, group=normal_group),  # leg4
    Circle(radius=10, color=color3, batch=batch_b, group=cat_face_group),  # thigh1
    Circle(radius=10, color=color3, batch=batch_b, group=normal_group),  # thigh2
    Circle(radius=10, color=color3, batch=batch_b, group=normal_group),  # thigh3
    Circle(radius=10, color=color3, batch=batch_b, group=normal_group),  # thigh4
    Circle(radius=10, color=color4, batch=batch_b, group=cat_face_group),  # paw1
    Circle(radius=10, color=color4, batch=batch_b, group=normal_group),  # paw2
    Circle(radius=10, color=color4, batch=batch_b, group=normal_group),  # paw3
    Circle(radius=10, color=color4, batch=batch_b, group=normal_group),  # paw4
    Circle(radius=20, color=color1, batch=batch_b, group=cat_face_group),  # ear1
    Circle(radius=20, color=color1, batch=batch_b, group=cat_face_group),  # ear2
    Circle(radius=20, color=color3, batch=batch_b, group=cat_face_group),  # tail1
    Circle(radius=20, color=color3, batch=batch_b, group=normal_group),  # tail2
    Circle(radius=20, color=color4, batch=batch_b, group=normal_group),  # tail3
    Circle(radius=10, color=black, batch=batch_b, group=cat_face_group),  # eye_left
    Circle(radius=10, color=black, batch=batch_b, group=cat_face_group),  # eye_right
    Circle(radius=5, color=color4, batch=batch_b, group=cat_face_group),  # nose
]


cat_body_part_index: list[tuple[int, int, int]] = [
    (0, 0, -1),  # Main body (0)
    (-50, 120, 0),  # Head   (1)
    (-60, -80, 0),  # Leg1   (2)
    (-30, -100, 0),  # Leg2  (3)
    (30, -100, 0),  # Leg3   (4)
    (60, -80, 0),  # Leg4    (5)
    (-10, -15, 2),  # thigh1 (6)
    (0, -20, 3),  # thigh2   (7)
    (5, -20, 4),  # thigh3   (8)
    (10, -15, 5),  # thigh4  (9)
    (-10, -15, 6),  # paw1  (10)
    (0, -20, 7),  # paw2    (11)
    (5, -20, 8),  # paw3    (12)
    (10, -15, 9),  # paw4   (13)
    (-30, 40, 1),  # ear1   (14)
    (30, 40, 1),  # ear2    (15)
    (105, 30, 0),  # tail1  (16)
    (30, 20, 16),  # tail2  (17)
    (30, 20, 17),  # tail3  (18)
    (-20, 0, 1),  # eye_left (19)
    (20, 0, 1),  # eye_left (20)
    (0, -10, 1),  # nose (21)
]

cat = Cat(cat_body_parts, cat_body_part_index)

wrap = WrapAroundScreen(circles, lines)


@window.event
def on_draw():
    window.clear()

    if current_state == "Start_state":
        wrap.start_screen()

    elif current_state == "A_state":
        batch_a.draw()

    elif current_state == "B_state":
        batch_b.draw()


def update(dt: float):
    global current_state

    if current_state == "Start_state":
        pass

    elif current_state == "A_state":
        wrap.update(dt)

    elif current_state == "B_state":
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


def on_key_release(symbol, modifer):
    global current_state

    if current_state == "Start_state":
        if symbol == key.ENTER:
            current_state = "A_state"

    elif current_state == "A_state":
        if symbol == key.SPACE:
            current_state = "B_state"

    elif current_state == "B_state":
        if symbol == key.SPACE:
            current_state = "A_state"


# for the keys that is to be pressed once
window.push_handlers(on_key_release)
# Schedule updates
pyglet.clock.schedule_interval(update, dt)

# Run the app
pyglet.app.run()
