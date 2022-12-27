import numpy as np

NUM_POINTS = 100000
NUM_DIMENSIONS = 3

positions = np.random.uniform(low=-15.0, high = 15.0, size=(NUM_POINTS, NUM_DIMENSIONS))

positions.astype('float32').tofile('data.dat')