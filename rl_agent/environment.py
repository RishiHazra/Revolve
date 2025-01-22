import numpy as np


class CustomEnvironment:
    def __init__(self):
        self.observation = None  # Initialize observation as None

    def update_state(self, observation):
        self.observation = observation  # Directly store the observation

    @property
    def env_state(self):
        return {"observation": self.observation}
