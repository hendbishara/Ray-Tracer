import numpy as np
from intersection import Intersection


class Sphere():
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index
    
    """calculates ray intersection with sphere"""
    def intersect(self, ray):
        center=self.position
        oc=ray.origin - center
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, oc)
        c = np.dot(oc,oc) - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        #when delta>0 there are two points of intersection
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
                return Intersection(self, ray, t)
        return None
