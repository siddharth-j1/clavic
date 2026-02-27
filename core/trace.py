# core/trace.py

import numpy as np


class Trace:
    def __init__(self, time, position, orientation=None, velocity=None, gains=None):
        self.time = np.array(time)
        self.position = np.array(position)
        self.orientation = orientation
        self.velocity = velocity
        self.gains = gains