

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = position
        self.look_at = look_at
        self.up_vector = up_vector
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.direction=None
        self.right=None

    def fix(self,direction,right,up_vector):
        self.direction=direction
        self.right=right
        self.up_vector=up_vector
