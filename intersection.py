class Intersection:
    def __init__(self, shape, ray, t):
        self.ray = ray
        self.shape = shape
        self.t = t
        self.point = ray.origin + t * ray.direction

