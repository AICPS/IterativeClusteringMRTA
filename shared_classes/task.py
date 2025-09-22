import numpy as np

class Task:
    def __init__(self, id, reward_matrix, x, y, kappa):
        self.reward_matrix = np.array(reward_matrix)
        self.x = x
        self.y = y
        self.id = id
        self.kappa = kappa
        self.capability_vector = self.compute_capability_vector()

    def compute_capability_vector(self):
        capability_vector = np.zeros(self.kappa, dtype=int)
        for i in range(self.kappa):
            # Check if there's any positive value in the reward matrix for this capability
            slices = [slice(None)] * self.reward_matrix.ndim
            slices[i] = slice(1, None)  # Exclude the first element (index 0)
            if np.any(self.reward_matrix[tuple(slices)] > 0):
                capability_vector[i] = 1
        return capability_vector

    def get_id(self):
        return self.id

    def get_reward_matrix(self):
        return self.reward_matrix

    def get_dimensions(self):
        return self.reward_matrix.shape

    def get_reward(self, *indices):
        try:
            return self.reward_matrix[indices]
        except IndexError:
            raise IndexError("Indices out of bounds for the reward matrix.")

    def set_reward(self, value, *indices):
        try:
            self.reward_matrix[indices] = value
            # Update capability vector if necessary
            self.capability_vector = self.compute_capability_vector()
        except IndexError:
            raise IndexError("Indices out of bounds for the reward matrix.")

    def get_location(self):
        return (self.x, self.y)
    
    def set_location(self, x, y):
        self.x = x
        self.y = y

    def get_capability_vector(self):
        return self.capability_vector

    def __str__(self):
        return (f"Task ID: {self.id}\n"
                f"Reward matrix:\n{self.reward_matrix}\n"
                f"Capability vector: {self.capability_vector}\n"
                f"Location: ({self.x}, {self.y})")