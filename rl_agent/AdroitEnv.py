"""An Adroit arm environment with door task using the Gymnasium API.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""

import inspect
import os
import json
from os import path
from typing import Optional
from typing import Callable, List, Tuple, Optional, Dict
import torch
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle
from rl_agent.environment import CustomEnvironment

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 90.0,
}


def define_function_from_string(
    function_string: str,
) -> Tuple[Optional[Callable], List[str]]:
    namespace = {}
    additional_globals = {
        "torch": torch,
        "np": np,
        "Tuple": Tuple,
        "List": List,
        "Callable": Callable,
        "Optional": Optional,
        "Dict": Dict,
    }
    namespace.update(additional_globals)
    exec(function_string, namespace)
    function = next(
        (value for key, value in namespace.items() if key == "compute_reward"), None
    )
    args = inspect.getfullargspec(function).args if function else []
    return function, args


def call_reward_func_dynamically(reward_func, env_state):
    params = inspect.signature(reward_func).parameters
    args_to_pass = {param: env_state[param] for param in params if param in env_state}
    reward, reward_components = reward_func(**args_to_pass)
    return reward, reward_components


class AdroitHandDoorEnv(MujocoEnv, EzPickle):
    """
    ## Description

    This environment was introduced in ["Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations"](https://arxiv.org/abs/1709.10087)
    by Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine.

    The environment is based on the [Adroit manipulation platform](https://github.com/vikashplus/Adroit), a 28 degree of freedom system which consists of a 24 degrees of freedom
    ShadowHand and a 4 degree of freedom arm. The task to be completed consists on undoing the latch and swing the door open.
    The latch has significant dry friction and a bias torque that forces the door to stay closed. Agent leverages environmental interaction to develop the understanding of the latch
    as no information about the latch is explicitly provided. The position of the door is randomized. Task is considered complete when the door touches the door stopper at the other end.

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (28,), float32)`. The control actions are absolute angular positions of the Adroit hand joints. The input of the control actions is set to a range between -1 and 1 by scaling the real actuator angle ranges in radians.
    The elements of the action array are the following:

    | Num | Action                                                                                  | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
    | --- | --------------------------------------------------------------------------------------- | ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | Linear translation of the full arm towards the door                                     | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTz                           | slide | position (m)|
    | 1   | Angular up and down movement of the full arm                                            | -1          | 1           | -0.4 (rad)   | 0.25 (rad)  | A_ARRx                           | hinge | angle (rad) |
    | 2   | Angular left and right and down movement of the full arm                                | -1          | 1           | -0.3 (rad)   | 0.3 (rad)   | A_ARRy                           | hinge | angle (rad) |
    | 3   | Roll angular movement of the full arm                                                   | -1          | 1           | -1.0 (rad)   | 2.0 (rad)   | A_ARRz                           | hinge | angle (rad) |
    | 4   | Angular position of the horizontal wrist joint (radial/ulnar deviation)                 | -1          | 1           | -0.524 (rad) | 0.175 (rad) | A_WRJ1                           | hinge | angle (rad) |
    | 5   | Angular position of the horizontal wrist joint (flexion/extension)                      | -1          | 1           | -0.79 (rad)  | 0.61 (rad)  | A_WRJ0                           | hinge | angle (rad) |
    | 6   | Horizontal angular position of the MCP joint of the forefinger (adduction/abduction)    | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_FFJ3                           | hinge | angle (rad) |
    | 7   | Vertical angular position of the MCP joint of the forefinger (flexion/extension)        | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ2                           | hinge | angle (rad) |
    | 8   | Angular position of the PIP joint of the forefinger (flexion/extension)                 | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ1                           | hinge | angle (rad) |
    | 9   | Angular position of the DIP joint of the forefinger                                     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ0                           | hinge | angle (rad) |
    | 10  | Horizontal angular position of the MCP joint of the middle finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_MFJ3                           | hinge | angle (rad) |
    | 11  | Vertical angular position of the MCP joint of the middle finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ2                           | hinge | angle (rad) |
    | 12  | Angular position of the PIP joint of the middle finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ1                           | hinge | angle (rad) |
    | 13  | Angular position of the DIP joint of the middle finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ0                           | hinge | angle (rad) |
    | 14  | Horizontal angular position of the MCP joint of the ring finger (adduction/abduction)   | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_RFJ3                           | hinge | angle (rad) |
    | 15  | Vertical angular position of the MCP joint of the ring finger (flexion/extension)       | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ2                           | hinge | angle (rad) |
    | 16  | Angular position of the PIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ1                           | hinge | angle (rad) |
    | 17  | Angular position of the DIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ0                           | hinge | angle (rad) |
    | 18  | Angular position of the CMC joint of the little finger                                  | -1          | 1           | 0 (rad)      | 0.7(rad)    | A_LFJ4                           | hinge | angle (rad) |
    | 19  | Horizontal angular position of the MCP joint of the little finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_LFJ3                           | hinge | angle (rad) |
    | 20  | Vertical angular position of the MCP joint of the little finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ2                           | hinge | angle (rad) |
    | 21  | Angular position of the PIP joint of the little finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ1                           | hinge | angle (rad) |
    | 22  | Angular position of the DIP joint of the little finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ0                           | hinge | angle (rad) |
    | 23  | Horizontal angular position of the CMC joint of the thumb finger                        | -1          | 1           | -1.047 (rad) | 1.047 (rad) | A_THJ4                           | hinge | angle (rad) |
    | 24  | Vertical Angular position of the CMC joint of the thumb finger                          | -1          | 1           | 0 (rad)      | 1.3 (rad)   | A_THJ3                           | hinge | angle (rad) |
    | 25  | Horizontal angular position of the MCP joint of the thumb finger (adduction/abduction)  | -1          | 1           | -0.26 (rad)  | 0.26(rad)   | A_THJ2                           | hinge | angle (rad) |
    | 26  | Vertical angular position of the MCP joint of the thumb finger (flexion/extension)      | -1          | 1           | -0.52 (rad)  | 0.52 (rad)  | A_THJ1                           | hinge | angle (rad) |
    | 27  | Angular position of the IP joint of the thumb finger (flexion/extension)                | -1          | 1           | -1.571 (rad) | 0 (rad)     | A_THJ0                           | hinge | angle (rad) |


    ## Observation Space

    The observation space is of the type `Box(-inf, inf, (39,), float64)`. It contains information about the angular position of the finger joints, the pose of the palm of the hand, as well as state of the latch and door.

    | Num | Observation                                                                 | Min    | Max    | Joint Name (in corresponding XML file) | Site Name (in corresponding XML file) | Joint Type| Unit                     |
    |-----|-----------------------------------------------------------------------------|--------|--------|----------------------------------------|---------------------------------------|-----------|------------------------- |
    | 0   | Angular position of the vertical arm joint                                  | -Inf   | Inf    | ARRx                                   | -                                     | hinge     | angle (rad)              |
    | 1   | Angular position of the horizontal arm joint                                | -Inf   | Inf    | ARRy                                   | -                                     | hinge     | angle (rad)              |
    | 2   | Roll angular value of the arm                                               | -Inf   | Inf    | ARRz                                   | -                                     | hinge     | angle (rad)              |
    | 3   | Angular position of the horizontal wrist joint                              | -Inf   | Inf    | WRJ1                                   | -                                     | hinge     | angle (rad)              |
    | 4   | Angular position of the vertical wrist joint                                | -Inf   | Inf    | WRJ0                                   | -                                     | hinge     | angle (rad)              |
    | 5   | Horizontal angular position of the MCP joint of the forefinger              | -Inf   | Inf    | FFJ3                                   | -                                     | hinge     | angle (rad)              |
    | 6   | Vertical angular position of the MCP joint of the forefinge                 | -Inf   | Inf    | FFJ2                                   | -                                     | hinge     | angle (rad)              |
    | 7   | Angular position of the PIP joint of the forefinger                         | -Inf   | Inf    | FFJ1                                   | -                                     | hinge     | angle (rad)              |
    | 8   | Angular position of the DIP joint of the forefinger                         | -Inf   | Inf    | FFJ0                                   | -                                     | hinge     | angle (rad)              |
    | 9   | Horizontal angular position of the MCP joint of the middle finger           | -Inf   | Inf    | MFJ3                                   | -                                     | hinge     | angle (rad)              |
    | 10  | Vertical angular position of the MCP joint of the middle finger             | -Inf   | Inf    | MFJ2                                   | -                                     | hinge     | angle (rad)              |
    | 11  | Angular position of the PIP joint of the middle finger                      | -Inf   | Inf    | MFJ1                                   | -                                     | hinge     | angle (rad)              |
    | 12  | Angular position of the DIP joint of the middle finger                      | -Inf   | Inf    | MFJ0                                   | -                                     | hinge     | angle (rad)              |
    | 13  | Horizontal angular position of the MCP joint of the ring finger             | -Inf   | Inf    | RFJ3                                   | -                                     | hinge     | angle (rad)              |
    | 14  | Vertical angular position of the MCP joint of the ring finger               | -Inf   | Inf    | RFJ2                                   | -                                     | hinge     | angle (rad)              |
    | 15  | Angular position of the PIP joint of the ring finger                        | -Inf   | Inf    | RFJ1                                   | -                                     | hinge     | angle (rad)              |
    | 16  | Angular position of the DIP joint of the ring finger                        | -Inf   | Inf    | RFJ0                                   | -                                     | hinge     | angle (rad)              |
    | 17  | Angular position of the CMC joint of the little finger                      | -Inf   | Inf    | LFJ4                                   | -                                     | hinge     | angle (rad)              |
    | 18  | Horizontal angular position of the MCP joint of the little finger           | -Inf   | Inf    | LFJ3                                   | -                                     | hinge     | angle (rad)              |
    | 19  | Vertical angular position of the MCP joint of the little finger             | -Inf   | Inf    | LFJ2                                   | -                                     | hinge     | angle (rad)              |
    | 20  | Angular position of the PIP joint of the little finger                      | -Inf   | Inf    | LFJ1                                   | -                                     | hinge     | angle (rad)              |
    | 21  | Angular position of the DIP joint of the little finger                      | -Inf   | Inf    | LFJ0                                   | -                                     | hinge     | angle (rad)              |
    | 22  | Horizontal angular position of the CMC joint of the thumb finger            | -Inf   | Inf    | THJ4                                   | -                                     | hinge     | angle (rad)              |
    | 23  | Vertical Angular position of the CMC joint of the thumb finger              | -Inf   | Inf    | THJ3                                   | -                                     | hinge     | angle (rad)              |
    | 24  | Horizontal angular position of the MCP joint of the thumb finger            | -Inf   | Inf    | THJ2                                   | -                                     | hinge     | angle (rad)              |
    | 25  | Vertical angular position of the MCP joint of the thumb finger              | -Inf   | Inf    | THJ1                                   | -                                     | hinge     | angle (rad)              |
    | 26  | Angular position of the IP joint of the thumb finger                        | -Inf   | Inf    | THJ0                                   | -                                     | hinge     | angle (rad)              |
    | 27  | Angular position of the door latch                                          | -Inf   | Inf    | latch                                  | -                                     | hinge     | angle (rad)              |
    | 28  | Angular position of the door hinge                                           | -Inf   | Inf    | door_hinge                             | -                                     | hinge     | angular velocity (rad/s) |
    | 29  | Position of the center of the palm in the x direction                       | -Inf   | Inf    | -                                      | S_grasp                               | -         | position (m)             |
    | 30  | Position of the center of the palm in the y direction                       | -Inf   | Inf    | -                                      | S_grasp                               | -         | position (m)             |
    | 31  | Position of the center of the palm in the z direction                       | -Inf   | Inf    | -                                      | S_grasp                               | -         | position (m)             |
    | 32  | x position of the handle of the door                                        | -Inf   | Inf    | -                                      | S_handle                              | -         | position (m)             |
    | 33  | y position of the handle of the door                                        | -Inf   | Inf    | -                                      | S_handle                              | -         | position (m)             |
    | 34  | z position of the handle of the door                                        | -Inf   | Inf    | -                                      | S_handle                              | -         | position (m)             |
    | 35  | x positional difference from the palm of the hand to the door handle        | -Inf   | Inf    | -                                      | S_handle,S_grasp                      | -         | position (m)             |
    | 36  | y positional difference from the palm of the hand to the door handle        | -Inf   | Inf    | -                                      | S_handle,S_grasp                      | -         | position (m)             |
    | 37  | z positional difference from the palm of the hand to the door handle        | -Inf   | Inf    | -                                      | S_handle,S_grasp                      | -         | position (m)             |
    | 38  | 1 if the door is open, otherwise -1                                         | -1     | 1      | door_hinge                             | -                                     | hinge     | bool                     |

    ## Rewards

    The environment can be initialized in either a `dense` or `sparse` reward variant.

    In the `dense` reward setting, the environment returns a `dense` reward function that consists of the following parts:
    - `get_to_handle`: increasing negative reward the further away the palm of the hand is from the door handle. This is computed as the 3 dimensional Euclidean distance between both body frames.
        This penalty is scaled by a factor of `0.1` in the final reward.
    - `open_door`: squared error of the current door hinge angular position and the open door state. The final reward is scaled by `0.1`.
    - `velocity_penalty`: Minor velocity penalty for the full dynamics of the environments. Used to bound the velocity of the bodies in the environment.
        It equals the norm of all the joint velocities. This penalty is scaled by a factor of `0.00001` in the final reward.
    - `door_hinge_displacement`: adds a positive reward of `2` if the door hinge is opened more than `0.2` radians, `8` if more than `1.0` randians, and `10` if more than `1.35` radians.

    The `sparse` reward variant of the environment can be initialized by calling `gym.make('AdroitHandDoorSparse-v1')`.
    In this variant, the environment returns a reward of 10 for environment success and -0.1 otherwise.

    ## Starting State

    To add stochasticity to the environment the `(x,y,z)` coordinates of the door are randomly sampled each time the environment is reset. The values are extracted from a uniform distribution
    with ranges `[-0.3,-0.2]` for the `x` coordinate, `[0.25,0.35]` for the `y` coordinate, and `[0.252,0.35]` for the `z` coordinate.

    The joint values of the environment are deterministically initialized to a zero.

    For reproducibility, the starting state of the environment can also be set when calling `env.reset()` by passing the `options` dictionary argument (https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)
    with the `initial_state_dict` key. The `initial_state_dict` key must be a dictionary with the following items:

    * `qpos`: np.ndarray with shape `(30,)`, MuJoCo simulation joint positions
    * `qvel`: np.ndarray with shape `(30,)`, MuJoCo simulation joint velocities
    * `door_body_pos`: np.ndarray with shape `(3,)`, cartesian coordinates of the door body

    The state of the simulation can also be set at any step with the `env.set_env_state(initial_state_dict)` method.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 200 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 400 make the environment as follows:

    ```python
    import gymnasium as gym

    env = gym.make('AdroitHandDoor-v1', max_episode_steps=400)
    ```

    ## Version History

    * v1: refactor version of the D4RL environment, also create dependency on newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }
    # /home/alkis/.local/lib/python3.8/site-packages/gymnasium_robotics/envs/assets/adroit_hand/adroit_door.xml
    #    def __init__(self, reward_type: str = "dense", max_episode_steps: int = 400, mode: str = "train", **kwargs):

    def __init__(
        self,
        reward_fn_path: str,
        counter: int,
        iteration: int,
        group_id: str,
        llm_model: str,
        baseline: str,
        max_episode_steps: int = 400,
        mode: str = "train",
        **kwargs,
    ):
        # xml_file_path = path.join(
        #     path.dirname(path.realpath(__file__)),
        #     "../assets/adroit_hand/adroit_door.xml",
        # )
        # /home/alkis/.local/lib/python3.8/site-packages/gymnasium_robotics/envs/assets/adroit_hand/adroit_door.xml
        self.max_episode_steps = max_episode_steps
        self.reward_fn_path = reward_fn_path
        self.counter = counter
        self.iteration = iteration
        self.group_id = group_id
        self.llm_model = llm_model
        self.baseline = baseline
        self.custom_env = CustomEnvironment()

        self.base_path = os.path.join(
            os.environ["ROOT_PATH"],
            f"{baseline}/{llm_model}/group_{group_id}/reward_history",
        )
        self.filename = f"{self.iteration}_{self.counter}.json"
        self.filepath = os.path.join(
            self.base_path, self.filename
        )  # for reward components entries
        os.makedirs(self.base_path, exist_ok=True)
        self.filename2 = f"{self.iteration}_{self.counter}.txt"
        self.filepath2 = os.path.join(
            self.base_path, self.filename2
        )  # for wring episode steps and True False for calcualting fitness after
        self.filename3 = f"testing_{self.iteration}_{self.counter}.txt"
        self.filepath3 = os.path.join(self.base_path, self.filename3)

        self.filename4 = f"rewards_{self.iteration}_{self.counter}.txt"
        self.filepath4 = os.path.join(self.base_path, self.filename4)

        self.current_step = 0  # Initialize step counter
        xml_file_path = "/home/alkis/.local/lib/python3.8/site-packages/gymnasium_robotics/envs/assets/adroit_hand/adroit_door.xml"
        self.mode = mode
        reward_func_str = open(reward_fn_path, "r").read()
        self.reward_func, _ = define_function_from_string(reward_func_str)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.model)

        # Override action_space to -1, 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        )

        # change actuator sensitivity
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"]
            + 1,
            :3,
        ] = np.array([10, 0, 0])
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"]
            + 1,
            :3,
        ] = np.array([1, 0, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"]
            + 1,
            :3,
        ] = np.array([0, -10, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"]
            + 1,
            :3,
        ] = np.array([0, -1, 0])

        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        self.door_hinge_addrs = self.model.jnt_dofadr[
            self._model_names.joint_name2id["door_hinge"]
        ]
        self.grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.handle_site_id = self._model_names.site_name2id["S_handle"]
        self.door_body_id = self._model_names.body_name2id["frame"]

        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64
                ),
                "door_body_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )

        EzPickle.__init__(self, **kwargs)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        self.current_step += 1  # Increment step counter

        a = self.act_mean + a * self.act_rng  # mean center and scale

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        joint_velocities = self.data.qvel.ravel()
        joint_forces = self.data.actuator_force.ravel()
        self.custom_env.update_state(obs, joint_velocities, joint_forces)
        reward, reward_components = call_reward_func_dynamically(
            self.reward_func, self.custom_env.env_state
        )
        self.rewards.append(reward)
        for key, value in reward_components.items():
            if key not in self.reward_components_log:
                self.reward_components_log[key] = []
            self.reward_components_log[key].append(value)

        # compute the sparse reward variant first
        goal_distance = self.data.qpos[self.door_hinge_addrs]
        goal_achieved = True if goal_distance >= 1.35 else False

        episode_summary = {
            "total_reward": sum(self.rewards),
            "episode_components": {
                key: sum(values) for key, values in self.reward_components_log.items()
            },
        }

        with open(self.filepath4, "a") as file:
            file.write(f"Reward {reward}\n")

        done = (goal_distance > 1.35) or (self.current_step >= 400)
        if done and self.mode == "test":
            with open(self.filepath3, "a") as file:
                file.write(
                    f"Episode finished at step {self.current_step}: Success={goal_achieved}\n"
                )
            # with open(self.filepath4, "a") as file:
            #     file.write(f"Episodic Reward{self.current_step}:\n")

        # done = goal_distance > 1.35
        succ = goal_distance > 1.35
        # done = self.current_step >= self.max_episode_steps
        if done:
            with open(self.filepath, "a") as file:
                json.dump(episode_summary, file)
                file.write("\n")  # New line for each episode

            # Reset rewards and components for the next episode
            self.rewards = []
            self.reward_components_log = {key: [] for key in reward_components.keys()}

            with open(self.filepath2, "a") as file:
                file.write(f"Episode {self.current_step}: Success={succ}\n")

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, False, dict(success=goal_achieved)

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qpos = self.data.qpos.ravel()
        joint_velocities = self.data.qvel.ravel()
        joint_forces = (
            self.data.actuator_force.ravel()
        )  # Force/torque applied by each actuator

        handle_pos = self.data.site_xpos[self.handle_site_id].ravel()
        palm_pos = self.data.site_xpos[self.grasp_site_id].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_addrs]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qpos[-1]
        return np.concatenate(
            [
                qpos[1:-2],  # Existing joint positions
                [latch_pos],  # Existing latch position
                door_pos,  # Existing door position
                palm_pos,  # Existing palm position
                handle_pos,  # Existing handle position
                palm_pos - handle_pos,  # Existing positional difference
                [door_open],  # Existing door open indicator
                # joint_velocities,       # New: Joint velocities
                # joint_forces          # New: Forces/torques
            ]
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        obs, info = super().reset(seed=seed)
        if options is not None and "initial_state_dict" in options:
            self.set_env_state(options["initial_state_dict"])
            obs = self._get_obs()
        self.current_step = 0  # Reset step counter
        return obs, info

    def reset_model(self):
        self.rewards = []
        self.reward_components_log = {}
        self.model.body_pos[self.door_body_id, 0] = self.np_random.uniform(
            low=-0.3, high=-0.2
        )
        self.model.body_pos[self.door_body_id, 1] = self.np_random.uniform(
            low=0.25, high=0.35
        )
        self.model.body_pos[self.door_body_id, 2] = self.np_random.uniform(
            low=0.252, high=0.35
        )
        self.set_state(self.init_qpos, self.init_qvel)

        return self._get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_body_id].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        assert self._state_space.contains(
            state_dict
        ), f"The state dictionary {state_dict} must be a member of {self._state_space}."
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        self.model.body_pos[self.door_body_id] = state_dict["door_body_pos"]
        self.set_state(qp, qv)
