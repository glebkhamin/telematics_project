import numpy as np

num_particles = 100
width, height = 400, 300  # Simulation area dimensions

# Initialize particles randomly across the space
particles = np.random.rand(num_particles, 2) * [width, height]

# Define initial state for the car's actual position and velocity (magnitude and direction)
car_position = np.array([200, 150])
car_velocity = 5  # units per time step
car_direction = np.pi / 4  # 45 degrees in radians

# part of physical model
def move_car(position, velocity, direction, dt=1):
    """Predict the car's next position based on velocity and direction."""
    dx = velocity * np.cos(direction) * dt
    dy = velocity * np.sin(direction) * dt
    return position + np.array([dx, dy])


# part of filter
def predict_car_movement(position, velocity, direction, dt=1):
    """Predict the car's next position based on velocity and direction."""
    dx = velocity * np.cos(direction) * dt
    dy = velocity * np.sin(direction) * dt
    return position + np.array([dx, dy])

# part of the physical model
# Simulate a change in velocity and direction for realism
car_velocity += np.random.randn()  # Random change in velocity
car_direction += np.random.randn() * 0.1  # Random change in direction

# Predict car's new position
car_position = move_car(car_position, car_velocity, car_direction)

# part of the filter
# Predict particles' positions using a similar motion model, adding some noise
particle_velocities = np.random.normal(car_velocity, 1, num_particles)  # Velocity with noise
particle_directions = np.random.normal(car_direction, 0.1, num_particles)  # Direction with noise

for i in range(num_particles):
    particles[i] = predict_car_movement(particles[i], particle_velocities[i], particle_directions[i])


# plot on a graph all the particle positions and the car position as a red dot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.scatter(particles[:, 0], particles[:, 1], s=5)
plt.scatter(car_position[0], car_position[1], c="r")
plt.show()
