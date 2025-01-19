import numpy as np


class CustomEnvironment:
    def __init__(self):
        self.observation = None  # Initialize observation as None
        self.joint_velocities = None  # Initialize joint velocities as None
        self.joint_forces = None      # Initialize joint forces as None

    def update_state(self, observation, joint_velocities, joint_forces):
        self.observation = observation  # Directly store the observation
        self.joint_velocities = joint_velocities  # Update joint velocities
        self.joint_forces = joint_forces      # Update joint forces




    @property
    def env_state(self):
        return {
            'observation': self.observation,
            'joint_velocities': self.joint_velocities,
            'joint_forces': self.joint_forces
        }




                   