import numpy as np


class CustomEnvironment:
    def __init__(self):
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.collision = False
        self.min_pos = 10000.0
        self.speed = 0
        self.action_list = np.zeros(4)
        self.angular_velocity_x = 0.0
        self.angular_velocity_y = 0.0
        self.angular_velocity_z = 0.0
        self.episode_step_counter = 0
        self.total_step_counter = 0
        self.distance = 20

    @property
    def env_state(self):
        return self.__dict__
