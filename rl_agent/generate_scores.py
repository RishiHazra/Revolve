import inspect
import os
import time
from typing import Optional, Callable, List, Tuple, Dict

import airsim
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Input, Lambda
from tensorflow.keras.layers import GaussianNoise

# import tensorflow_addons as tfa

path = os.path.join(os.environ['ROOT_PATH'], 'rl_agent/gmaps3.txt')
data = np.loadtxt(path, delimiter=',')


def fitness_function(collision: bool, current_speed: float, min_distance_to_waypoint: float,
                     desired_speed_min: float = 9.0, desired_speed_max: float = 10.5, collision_penalty: float = -1,
                     speed_penalty: float = 0, speed_factor: float = 1.0, distance_factor: float = 1.0,
                     speed_tolerance: float = 2, max_distance_tolerance: float = 1) -> float:
    # Initialize fitness score
    fitness_score = 0.0

    # Check for collision
    if collision:
        return collision_penalty  # Return the collision penalty if a collision has occurred

    # Adjust desired speed range based on tolerance
    if current_speed < 2 or current_speed > 20:
        return speed_penalty

    if min_distance_to_waypoint > 4:
        return speed_penalty
    adjusted_min_speed = desired_speed_min - speed_tolerance
    adjusted_max_speed = desired_speed_max + speed_tolerance

    # Calculate speed reward
    if adjusted_min_speed <= current_speed <= adjusted_max_speed:
        speed_reward = speed_factor  # Max reward if within adjusted speed range
    else:
        # Penalize deviation from adjusted speed range, ensuring R_speed does not go below 0
        deviation = min(abs(current_speed - adjusted_min_speed), abs(current_speed - adjusted_max_speed))
        speed_reward = max(speed_factor * (1 - deviation / (adjusted_max_speed - adjusted_min_speed)), 0)

    if min_distance_to_waypoint <= max_distance_tolerance:
        distance_reward = 1
    else:
        excess_distance = min_distance_to_waypoint - max_distance_tolerance
        distance_penalty = max(0, distance_factor * (1 - excess_distance / max_distance_tolerance))
        distance_reward = distance_penalty

    total_reward = (speed_reward + distance_reward) / (speed_factor + distance_factor)

    print("speed,speed_reward", current_speed, speed_reward)
    print("min_distance_to_waypoint,distance_reward", min_distance_to_waypoint, distance_reward)
    print("total score", total_reward)
    print("\n \n")
    # Normalize fitness score to be between 0 and 1
    # Since R_speed is capped at 0 and speed_factor is the maximum, normalization is straightforward
    # fitness_score_normalized = fitness_score / speed_factor

    return total_reward


def read_reward_from_file(file_path: str) -> str:
    """
    Reads the reward function string from the specified file.

    Parameters:
    file_path (str): The path to the file containing the reward function string.

    Returns:
    str: The content of the file (reward function string).
    """
    with open(file_path, "r") as infile:
        return infile.read()


def define_function_from_string(function_string: str) -> Tuple[Optional[Callable], List[str]]:
    """
    Takes a string containing a function definition and returns the defined function.

    Args:
    - function_string (str): The string containing the function definition.

    Returns:
    - function: The defined function.
    """
    namespace = {}
    # TODO: add more additional globals?
    additional_globals = {'tf': tf, 'np': np, 'Tuple': Tuple,
                          'List': List, 'Callable': Callable,
                          'Optional': Optional, 'Dict': Dict}
    namespace.update(additional_globals)
    exec(function_string, namespace)
    # TODO: change 'compute_reward' to some other identifier
    function = next((value for key, value in namespace.items() if key == 'compute_reward'), None)
    args = inspect.getfullargspec(function).args if function else []
    return function, args


class CustomEnvironment:
    def __init__(self):
        self.curr_x = 0.
        self.curr_y = 0.
        self.vx = 0.
        self.vy = 0.
        self.yaw = 0.
        self.pitch = 0.
        self.collision = False
        self.min_pos = 10000.
        self.speed = 0
        self.action_list = np.zeros(4)
        self.angular_velocity_x = 0.
        self.angular_velocity_y = 0.
        self.angular_velocity_z = 0.
        self.episode_step_counter = 0
        self.total_step_counter = 0
        self.distance = 20

    @property
    def env_state(self):
        return self.__dict__


def eucl_dis(x, y, stx, sty):
    point1 = np.array([x, y])
    point2 = np.array([stx, sty])

    # Calculate the Euclidean distance between the points
    distance = np.linalg.norm(point1 - point2)
    return distance


file_path = os.path.join(os.environ['ROOT_PATH'], rl_agent / gmaps3.txt)


def get_min_pos(curr_x, curr_y):
    min_dis = 10000
    counter_pos = 0
    for i in data:  # data is a  list containing all recorded positions of the middle points of the road
        counter_pos = counter_pos + 1
        stx = i[0]
        sty = i[1]
        dis = eucl_dis(stx, sty, curr_x, curr_y)
        if (dis < min_dis):
            min_dis = dis
    min_dis = min_dis - 0.25
    min_dis = max(min_dis, 0)
    return min_dis


def call_reward_func_dynamically(reward_func, env_state):
    params = inspect.signature(reward_func).parameters
    args_to_pass = {param: env_state[param] for param in params if param in env_state}
    reward, reward_components = reward_func(**args_to_pass)
    return reward, reward_components


class DDDQN(tf.keras.Model):
    def __init__(self):
        super(DDDQN, self).__init__()
        self.activation = 'swish'
        self.dropout = 0.
        self.initializer = HeNormal(seed=None)

    def create_model(self):
        input_ = Input(shape=(100, 256, 12))
        activation = self.activation
        dropout = self.dropout
        initializer = self.initializer

        optimizer = tf.keras.optimizers.Adam(lr=0.0005, clipnorm=1.5)

        x = Lambda(lambda x: x / 255.0)(input_)
        x = GaussianNoise(stddev=0)(x)
        x = Conv2D(16, (5, 5), padding='same', activation=activation, name='Conv1')(x)
        x = BatchNormalization(name='bn1')(x)

        x = Conv2D(32, (3, 3), padding='same', activation=activation, name='Conv2')(x)
        x = BatchNormalization(name='bn2')(x)

        x = Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation=activation, name='Conv3')(x)
        x = BatchNormalization(name='bn3')(x)

        x = Conv2D(128, (3, 3), padding='same', strides=(2, 2), activation=activation, name='Conv4')(x)
        x = BatchNormalization(name='bn4')(x)

        x = Conv2D(256, (3, 3), padding='same', strides=(2, 2), activation=activation, name='Conv5')(x)
        x = BatchNormalization(name='bn5')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = Flatten()(x)
        x = layers.Dense(256, activation=activation, name='dense1')(x)
        x = BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

        input2 = Input(shape=(6))  # 2 euleor angles vx,vy, absolute speed
        input2_processed = Dense(256, kernel_initializer=initializer, activation=activation)(input2)
        x2 = BatchNormalization()(input2_processed)
        x2 = layers.Dropout(dropout)(x2)

        x = layers.Concatenate()([x, x2])

        value = layers.Dense(128, kernel_initializer=initializer, activation=activation)(x)
        value = tf.keras.layers.LayerNormalization()(value)
        value = layers.Dropout(dropout)(value)
        value = layers.Dense(1, kernel_initializer=initializer, activation="linear")(value)

        # Advantage stream
        advantage = layers.Dense(128, kernel_initializer=initializer, activation=activation, name='name1')(x)
        advantage = tf.keras.layers.LayerNormalization()(advantage)
        advantage = layers.Dropout(dropout, name='name3')(advantage)
        advantage = layers.Dense(66, kernel_initializer=initializer, activation="linear", name='name4')(advantage)

        # Combine value and advantage to get Q-values
        outputs = layers.Add()([value, layers.Subtract()(
            [advantage, tf.reduce_mean(advantage, axis=1, keepdims=True)])])  # add layer norm before q values

        model = tf.keras.Model([input_, input2], outputs)
        # model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=0.25), metrics=['accuracy'])  # can be
        # tested
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE)

        return model


import os
import random

import numpy as np
import tensorflow as tf

avail_actions = np.arange(-0.8, 0.85, 0.05)
avail_throttle = np.array([0, 1])
action_grid, throttle_grid = np.meshgrid(avail_actions, avail_throttle)
combined_array = np.array([action_grid.flatten(), throttle_grid.flatten()]).T
avail_actions_comb = np.round(combined_array, 2)


class DrivingAgent:
    def __init__(self, no_model, iteration, group_id, baseline):
        # self.no_model = no_model
        # for key, value in train_cfg.items():
        #  setattr(self, key, value)
        self.ddqn = DDDQN()
        self.q_net = self.ddqn.create_model()
        self.q_net2 = self.ddqn.create_model()
        self.iteration = iteration
        self.clipnorm = 1.5
        self.lr = 0.0002
        self.group_id = group_id
        self.tau = 0.0075
        self.epsilon_decay = 0.00015  # after 10k steps for 0.0001
        self.epsilon = 1
        self.batch_size = 32
        self.trainstep = 0
        self.min_epsilon = 0.01
        self.gamma = 0.99
        self.no_model = no_model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=self.clipnorm)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=self.clipnorm)
        self.target_net = self.ddqn.create_model()
        self.baseline = baseline
        #  self.weights_dir = os.path.join(os.environ['ROOT_PATH'], f'database/group_id_{group_id}')
        # os.makedirs(self.weights_dir, exist_ok=True)  # Ensure the weights directory exists

        loop_num = 1
        if loop_num != 0:
            self.epsilon = 0.
            self.q_net.load_weights(f"{os.environ['ROOT_PATH']}/{baseline}_database/gpt-4/group_{group_id}/"
                                    f"model_checkpoints/main1_{iteration}_{self.no_model}.h5")
        # self.load_network()
        else:
            self.update_target(tau=1)
            self.epsilon = 1
        print("starting epsilon value", self.epsilon)

    def act(self, state, state2):

        if np.random.rand() <= self.epsilon:
            a = random.randint(0, 10)  # inscresea probability for going straight
            if a > 6:
                b = 49
            else:
                b = np.random.choice([i for i in range(len(avail_actions_comb))])
            return b
        else:
            q_values = self.q_net([np.expand_dims(state, axis=0), np.expand_dims(state2, axis=0)])
            action = np.argmax(q_values)
            # print("q values",q_values)
            return action

    def construct_weights_path(self, base_filename):
        return os.path.join(self.weights_dir, f'{base_filename}_{self.iteration}_{self.no_model}.h5')

    def update_target(self, tau=None):
        if tau is None:
            tau = self.tau
        weights_q_net = self.q_net.get_weights()
        weights_target_net = self.target_net.get_weights()

        for i in range(len(weights_q_net)):
            weights_target_net[i] = self.tau * weights_q_net[i] + (1 - self.tau) * weights_target_net[i]

        self.target_net.set_weights(weights_target_net)

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def save_network(self):
        main_weights_file = self.construct_weights_path('main1')
        main_weights_file2 = self.construct_weights_path('main2')
        target_weights_file = self.construct_weights_path('target')

        self.q_net.save_weights(main_weights_file)
        self.q_net2.save_weights(main_weights_file2)
        self.target_net.save_weights(target_weights_file)

    def load_network(self):
        main_weights_file = self.construct_weights_path('main1')
        main_weights_file2 = self.construct_weights_path('main2')
        target_weights_file = self.construct_weights_path('target')

        if os.path.exists(main_weights_file):
            self.q_net.load_weights(main_weights_file)
        else:
            print(f"Weight file {main_weights_file} not found.")

        if os.path.exists(main_weights_file2):
            self.q_net2.load_weights(main_weights_file2)
        else:
            print(f"Weight file {main_weights_file2} not found.")

        if os.path.exists(target_weights_file):
            self.target_net.load_weights(target_weights_file)
        else:
            print(f"Weight file {target_weights_file} not found.")


def generate_behaviour(reward_func_path, counter_model, iteration, group_id, baseline):
    #  base_dir = os.path.join(os.environ['ROOT_PATH'], f"database/group_id_{group_id}")
    # os.makedirs(base_dir, exist_ok=True)

    agentoo7 = DrivingAgent(counter_model, iteration, group_id,
                            baseline)  # def __init__(self, no_model, iteration, group_id):

    reward_func_str = open(reward_func_path, 'r').read()
    reward_func, _ = define_function_from_string(reward_func_str)
    env_state = CustomEnvironment().env_state
    client = airsim.CarClient()
    # print("trying to connect on ip, port",ip_address, current_port)

    client.confirmConnection()
    client.enableApiControl(True)
    client.reset()
    car_controls = airsim.CarControls()

    ep = 0
    episode_step_counter = 0
    total_step_counter = 0
    rewards_history = {"total_reward": []}
    # recordings_dir = "path/to/AirSim/recordings/directory"  # Update this to your AirSim recordings directory
    total_fitness = 0
    total_fitness_steps = 0
    episodic_steps = 0

    # client.startRecording()
    max_steps = 100

    while total_step_counter <= max_steps:

        # startRecording()
        episode_start_time = time.time()  # Record the start time of the episode
        episode_total_reward = 0
        car_controls.brake = 0
        client.setCarControls(car_controls)
        episode_component_sums = {}

        done = False

        client.reset()
        done = 0

        orientation_scenario = random.choice(['straight', '90_left', '180_behind'])
        car_pose = client.simGetVehiclePose()
        if orientation_scenario == '90_left':
            car_pose.orientation = airsim.to_quaternion(0, 0, np.radians(-90))  # Yaw 90 degrees to the left
        elif orientation_scenario == '90_right':
            car_pose.orientation = airsim.to_quaternion(0, 0, np.radians(90))  # Yaw 90 degrees to the right
        elif orientation_scenario == '180_behind':
            car_pose.orientation = airsim.to_quaternion(0, 0, np.radians(185))  # Yaw 180 degrees
        #   car_pose.orientation = airsim.to_quaternion(0, 0, np.radians(185))
        client.simSetVehiclePose(car_pose, True)

        car_controls.throttle = 1
        car_controls.steering = 0
        client.setCarControls(car_controls)
        time.sleep(5.5)
        car_controls.throttle = 0

        car_controls.steering = 0
        client.setCarControls(car_controls)
        image_response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
        image_rgb = image1d.reshape(image_response.height, image_response.width, 3)

        frame = image_rgb[44:144, 0:256, 0:3]

        state = np.concatenate([frame, frame, frame, frame], axis=-1)

        position_info = client.getCarState()
        orientation = position_info.kinematics_estimated.orientation
        quaternion = np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])
        euler_angles = R.from_quat(quaternion).as_euler('zyx', degrees=True)
        linear_velocity = position_info.kinematics_estimated.linear_velocity
        vx, vy = linear_velocity.x_val, linear_velocity.y_val
        speed = float("{:.2f}".format(position_info.speed))
        yaw, pitch = euler_angles[0], euler_angles[1]
        car_info = np.array([yaw, pitch, vx, vy, speed, car_controls.steering])
        state2 = car_info
        counter_action = 0
        episode_step_counter = 0
        stop_counter = 0

        while not done:
            min_dis = 100000
            counter_action = counter_action + 1
            episode_step_counter = episode_step_counter + 1

            total_step_counter = total_step_counter + 1

            position_info = client.getCarState()

            speed = float("{:.2f}".format(position_info.speed))
            if speed < 0:
                speed = 0

            # env.render()
            #   client.simPause(True)
            action = agentoo7.act(state, state2)
            #  client.simPause(False)
            steering, throttle = avail_actions_comb[action][0], avail_actions_comb[action][1]
            #  print("throttle, steering",throttle, steering)
            # steering = avail_actions[action]

            car_controls.steering = steering

            car_controls.throttle = throttle
            # car_controls.throttle = throttle

            client.setCarControls(car_controls)
            image_response = \
                client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
            image_response = \
                client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
            image_response = \
                client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
            image_response = \
                client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]

            image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
            image_rgb = image1d.reshape(image_response.height, image_response.width, 3)

            new_frame = image_rgb[44:144, 0:256, 0:3]

            next_state = np.concatenate([state[:, :, 3:], new_frame], axis=-1)

            position_info = client.getCarState()
            collision_info = client.simGetCollisionInfo()
            orientation = position_info.kinematics_estimated.orientation
            quaternion = np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])
            euler_angles = R.from_quat(quaternion).as_euler('zyx', degrees=True)
            linear_velocity = position_info.kinematics_estimated.linear_velocity
            vx, vy = linear_velocity.x_val, linear_velocity.y_val
            speed = float("{:.2f}".format(position_info.speed))
            angular_velocity_x, angular_velocity_y, angular_velocity_z = position_info.kinematics_estimated.angular_velocity.x_val, position_info.kinematics_estimated.angular_velocity.y_val, position_info.kinematics_estimated.angular_velocity.z_val
            distance_data = client.getDistanceSensorData(vehicle_name="Car")
            distance = distance_data.distance  # The distance measured by the sensor in meters
            if (abs(distance - 20) <= 1.5):
                distance = 20
            yaw, pitch = euler_angles[0], euler_angles[1]
            car_info = np.array([yaw, pitch, vx, vy, speed, car_controls.steering])
            next_state2 = car_info
            min_pos = get_min_pos(
                position_info.kinematics_estimated.position.x_val, position_info.kinematics_estimated.position.y_val)

            env_state['curr_x'] = position_info.kinematics_estimated.position.x_val
            env_state['curr_y'] = position_info.kinematics_estimated.position.y_val
            env_state['vx'] = vx  # Ensure this matches your intended key, changed from 'curr_vx' for consistency
            env_state['vy'] = vy
            env_state['yaw'] = euler_angles[0]
            env_state['pitch'] = euler_angles[1]
            env_state['speed'] = speed
            env_state['collision'] = collision_info.has_collided
            env_state['min_pos'] = min_pos
            env_state['angular_velocity_x'] = angular_velocity_x
            env_state['angular_velocity_y'] = angular_velocity_y
            env_state['angular_velocity_z'] = angular_velocity_z
            env_state['total_step_counter'] = total_step_counter
            env_state['episode_step_counter'] = episode_step_counter
            env_state['distance'] = distance

            # Handling action_list
            if counter_action > 1:  # Ensure this counter is defined and updated in your loop
                env_state['action_list'][:-1] = env_state['action_list'][1:]  # Shift existing actions left
                env_state['action_list'][-1] = steering  # Add the new action to the end
            else:
                env_state['action_list'] = np.array([steering] * 4)  # Initialize actio

            reward, reward_components = call_reward_func_dynamically(reward_func, env_state)
            if (episodic_steps > 5):
                step_fitness = fitness_function(collision=collision_info.has_collided, current_speed=speed,
                                                min_distance_to_waypoint=min_pos, speed_tolerance=1.5)
                total_fitness += step_fitness
                total_fitness_steps = total_fitness_steps + 1
            #  def fitness_function(collision: bool, current_speed: float, min_distance_to_waypoint: float, desired_speed_min: float = 9.0, desired_speed_max: float = 10.5, collision_penalty: float = 0, speed_factor: float = 1.0, distance_factor: float = 1.0, speed_tolerance: float = 1.5, max_distance_tolerance: float = 10.0) -> float:

            episodic_steps = episodic_steps + 1
            if (speed < 2):
                stop_counter = stop_counter + 1
                if (stop_counter > 10):
                    done = 1
                    # stop_counter=0
            else:
                stop_counter = 0

            if collision_info.has_collided == 1:
                #    print("collision")
                done = 1
                car_controls.throttle = 0
                car_controls.brake = 50
                car_controls.steering = 0
                time.sleep(1)
                client.setCarControls(car_controls)

            state = next_state
            episode_total_reward += reward
            for component, value in reward_components.items():
                if component not in episode_component_sums:
                    episode_component_sums[component] = value
                else:
                    episode_component_sums[component] += value

        print("final episode reward , with epsilon probability and total steps ", episode_total_reward,
              agentoo7.epsilon, total_step_counter)

        rewards_history["total_reward"].append(episode_total_reward)
        for component, sum_value in episode_component_sums.items():
            if component not in rewards_history:
                rewards_history[component] = [sum_value]
            else:
                rewards_history[component].append(sum_value)

        episode_step_counter = 0
        ep = ep + 1
    # client.stopRecording()
    return total_fitness / total_fitness_steps


############## best group 4 model 3_39


iteration = 2
group_id = 4
model_number = 100
baseline = 'revolve'
print("model number ", iteration, group_id, model_number)
rew_path = os.path.join(os.environ['ROOT_PATH'],
                        f'{baseline}_database/gpt-4/group_{group_id}/reward_fns/{iteration}_{model_number}.txt')
average_fitness = generate_behaviour(rew_path, counter_model=model_number, iteration=iteration, group_id=group_id,
                                     baseline=baseline)  # def run_training( reward_func_path, counter_model, iteration, group_id):
average_fitness = max(0, average_fitness)
print("avg fitness", average_fitness)
saving = 0
if saving == 1:
    file_path = os.path.join(os.environ['ROOT_PATH'], f'{baseline}_database/gpt-4/group_{group_id}/'
                                                      f'fitness_scores_auto/{iteration}_{model_number}.txt')
    directory = os.path.dirname(file_path)

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Open the file in write mode ('w') and write the average_fitness
    with open(file_path, 'w') as file:
        file.write(str(average_fitness))

#  if speed<5.5 :
#        throttle=1
#        elif speed>13:
#             throttle=0
