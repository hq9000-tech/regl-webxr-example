import numpy as np

NUM_POINTS = 1000
NUM_DIMENSIONS = 3

positions = np.random.uniform(low=-5.0, high = 5.0, size=(NUM_POINTS, NUM_DIMENSIONS))

positions.astype('float32').tofile('data.dat')