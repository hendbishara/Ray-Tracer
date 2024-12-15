from intersection import Intersection
import numpy as np

class Cube():
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index


    """calculate ray intersection with cube"""
    def intersect(self,ray):
        # Calculate min and max corners of the cube
        half_length = self.scale / 2
        center=self.position
        cube_min = (center[0] - half_length,center[1] - half_length,center[2] - half_length,)
        cube_max = (center[0] + half_length,center[1] + half_length,center[2] + half_length,)

        # Initialize t_min and t_max for all three slabs
        t_min = -np.inf
        t_max = np.inf

        # Check each slab
        for i in range(3):
            if ray.direction[i] != 0:  # Avoid division by zero
                t1 = (cube_min[i] - ray.origin[i]) / ray.direction[i]
                t2 = (cube_max[i] - ray.origin[i]) / ray.direction[i]

                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max:
                    return None

            elif ray.origin[i] <= cube_min[i] or ray.origin[i] >= cube_max[i]:
                # The ray is parallel to the slab planes and outside the slab
                return None

        # Return the intersection point, t_min should be the entry point
        if t_min > 0 and t_max > 0:
            return Intersection(self,ray,t_min)
        else:
            return None
