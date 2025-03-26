import numpy as np

COEFFICIENT_OF_RESTITUTION = 0.9  # Adjust as needed


def circle_collision(self, other: "Circle") -> None:
    # Calculate distance between circle centers
    distance = np.linalg.norm(self.position - other.position)

    # Avoid division by zero (exact overlap)
    if distance == 0:
        return

    # Compute the collision normal (unit vector)
    collision_normal = (other.position - self.position) / distance

    # Compute relative velocity
    relative_velocity = self.velocity - other.velocity

    # Project relative velocity onto collision normal
    velocity_along_normal = np.dot(relative_velocity, collision_normal)

    # Skip if circles are moving apart
    if velocity_along_normal > 0:
        return

    # Compute impulse scalar using impulse formula from notes
    reduced_mass = (self.mass * other.mass) / (self.mass + other.mass)
    impulse_magnitude = (
        (1 + COEFFICIENT_OF_RESTITUTION) * reduced_mass * (-velocity_along_normal)
    )

    # Compute impulse vector
    impulse = impulse_magnitude * collision_normal

    # Apply impulse to velocities
    self.velocity += impulse / self.mass
    other.velocity -= impulse / other.mass

    # Resolve overlap by moving circles apart
    overlap_distance = self.radius + other.radius - distance

    if overlap_distance > 0:
        # Distribute displacement based on mass ratio
        total_mass = self.mass + other.mass
        self_displacement_ratio = other.mass / total_mass
        other_displacement_ratio = self.mass / total_mass

        self.position -= collision_normal * (overlap_distance * self_displacement_ratio)
        other.position += collision_normal * (
            overlap_distance * other_displacement_ratio
        )
