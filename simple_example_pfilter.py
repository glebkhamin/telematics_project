import numpy as np

num_particles = 100
width, height = 400, 300  # Simulation area dimensions
particles = np.random.rand(num_particles, 2) * [width, height]

# Initialize car's actual position and velocity
car_position = np.array([200, 150])
car_velocity = np.array([5, -2])  # Car moves 5 units right and 2 units up

# Update the car's actual position
car_position += car_velocity

# Predict particles' new positions based on estimated or last known velocity, adding noise
estimated_velocity = car_velocity + np.random.randn(2) * 1  # Adding some noise for uncertainty
particles += estimated_velocity + np.random.randn(num_particles, 2) * 1

observation_noise = 10
observed_position = car_position + np.random.randn(2) * observation_noise

distances = np.linalg.norm(particles - observed_position, axis=1)
weights = 1.0 / (distances + 1.0)
weights /= np.sum(weights)

indices = np.random.choice(range(num_particles), size=num_particles, p=weights)
particles = particles[indices]

estimated_position = np.mean(particles, axis=0)
