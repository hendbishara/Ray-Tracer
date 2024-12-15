import numpy as np
from intersection import Intersection


class InfinitePlane():
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index

    """intersect ray with plane"""
    def intersect(self, ray):
        normal=self.normal
        point_on_plane = normal * self.offset
        transformed_point = point_on_plane

        # Compute new offset as the dot product of the transformed normal and transformed point
        transformed_offset = np.dot(normal, transformed_point)
        # Calculate the denominator of the intersection formula
        denominator = np.dot(ray.direction, normal)
        
        # If denominator is zero, the ray is parallel to the plane
        if abs(denominator) < 1e-6:
            return None  # No intersection, or the ray is within the plane

        # Calculate the numerator of the intersection formula
        numerator = transformed_offset - np.dot(ray.origin, normal)

        # Calculate t
        t = numerator / denominator

        # Return t if it's a valid intersection point (t >= 0)
        if t>=0:
            return Intersection(self, ray, t)
        else: 
            return None

