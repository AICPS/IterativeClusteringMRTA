import numpy as np

class Robot:
    def __init__(self, id, capabilities, x, y):
        
        self.capabilities = np.array(capabilities)
        self.x = x
        self.y = y
        self.id = id
    
    def get_id(self):
        return self.id

    def get_location(self):
        return (self.x, self.y)

    def set_location(self, x, y):
        self.x = x
        self.y = y

    def get_capabilities(self):
        return self.capabilities

    def __str__(self):
        return f"Robot at ({self.x}, {self.y}) with capabilities: {self.capabilities}"