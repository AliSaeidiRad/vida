import numpy as np


class ClockwiseAngleDistance:
    def __init__(self, origin):
        self.origin = origin

    def __call__(self, point, refvec=[0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        # Vector between point and the origin: v = p - o
        vector = [point[0] - self.origin[0], point[1] - self.origin[1]]
        # Length of vector: ||v||
        lenvector = np.linalg.norm(vector[0] - vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -np.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = (
            refvec[1] * normalized[0] - refvec[0] * normalized[1]
        )  # x1*y2 - y1*x2
        angle = np.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to
        # subtract them from 2*pi (360 degrees)
        if angle < 0:
            return 2 * np.pi + angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance
        # should come first.
        return angle, lenvector
