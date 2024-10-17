import inspect
import json
import os
import random
import time

import airsim
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

from rl_agent.agent import DrivingAgent
from rl_agent.buffer import PrioritizedReplayBuffer
from rl_agent.environment import CustomEnvironment
from utils import define_function_from_string

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
path = os.path.join(os.environ['ROOT_PATH'], 'rl_agent/gmaps3.txt')
data = np.loadtxt(path, delimiter=',')

avail_actions = np.arange(-0.8, 0.85, 0.05)
avail_throttle = np.array([0, 1])
action_grid, throttle_grid = np.meshgrid(avail_actions, avail_throttle)
combined_array = np.array([action_grid.flatten(), throttle_grid.flatten()]).T
avail_actions_comb = np.round(combined_array, 2)


def append_rewards_as_txt(filename, rewards_history):
    with open(filename, 'a') as file:  # Open in append mode
        for key, values in rewards_history.items():
            line = f"{key}: "  # Start with the key followed by a colon and space
            line += ', '.join(
                [str(value) for value in values])  # Convert all list elements to strings and join with commas
            file.write(line + '\n')  # Write the line to the file and end with a newline character


def eucl_dis(x, y, stx, sty):
    point1 = np.array([x, y])
    point2 = np.array([stx, sty])

    # Calculate the Euclidean distance between the points
    distance = np.linalg.norm(point1 - point2)
    return distance


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
    return min_dis


def call_reward_func_dynamically(reward_func, env_state):
    params = inspect.signature(reward_func).parameters
    args_to_pass = {param: env_state[param] for param in params if param in env_state}
    reward, reward_components = reward_func(**args_to_pass)
    return reward, reward_components


def check_speed(x):  # varyign speed between 28-37km/h
    if 8 < x < 10.5:
        r = random.choice([0, 1])
    elif x < 8:
        r = 1
    else:
        r = 0
    return r


def save_rewards_as_json(filename, rewards_history):
    # Convert TensorFlow tensors to Python floats
    print("saving in path ", filename)
    converted_rewards_history = {}
    for key, values in rewards_history.items():
        if isinstance(values, list):
            converted_rewards_history[key] = [float(value) for value in values]
        else:
            converted_rewards_history[key] = float(values)

    # Save the converted rewards history as a JSON file
    with open(filename, 'w') as json_file:
        json.dump(converted_rewards_history, json_file, indent=4)


def run_training(ip_address, reward_func_path, counter_model, iteration, group_id, current_port, llm_model, baseline):
    base_dir = os.path.join(os.environ['ROOT_PATH'],
                            f"{baseline}_database/{llm_model}/group_{group_id}/reward_history")
    os.makedirs(base_dir, exist_ok=True)

    agentoo7 = DrivingAgent(counter_model, iteration, group_id, llm_model, baseline)
    # def __init__(self, no_model, iteration, group_id):

    reward_func_str = open(reward_func_path, 'r').read()
    reward_func, _ = define_function_from_string(reward_func_str)

    filename_suffix = f"{iteration}_{counter_model}"
    rewards_history_filename = os.path.join(base_dir, f"{filename_suffix}.json")
    episodic_steps_filename = os.path.join(base_dir, f"episodic_steps_{filename_suffix}.txt")
    total_rewards_filename = os.path.join(base_dir, f"total_rewards_filename_{filename_suffix}.txt")
    env_state = CustomEnvironment().env_state
    client = airsim.CarClient(ip=ip_address, port=current_port)
    # print("trying to connect on ip, port",ip_address, current_port)

    client.confirmConnection()
    print('Connected on ip', ip_address)
    client.enableApiControl(True)
    client.reset()
    car_controls = airsim.CarControls()

    client.enableApiControl(True)
    buffer = PrioritizedReplayBuffer()

    # agentoo7 = driving_agent
    batch_size = 32
    ep = 0
    episode_step_counter = 0
    total_step_counter = 0
    rewards_history = {"total_reward": []}

    while total_step_counter <= 150000:
        episode_total_reward = 0
        car_controls.brake = 0
        client.setCarControls(car_controls)
        episode_component_sums = {}

        done = False

        client.reset()
        done = 0

        orientation_scenario = random.choice(['straight', '90_left', '90_right', '180_behind'])
        car_pose = client.simGetVehiclePose()
        orientation_scenario = '180_behind'

        if orientation_scenario == '90_left':
            car_pose.orientation = airsim.to_quaternion(0, 0, np.radians(-90))  # Yaw 90 degrees to the left
        elif orientation_scenario == '90_right':
            car_pose.orientation = airsim.to_quaternion(0, 0, np.radians(90))  # Yaw 90 degrees to the right
        elif orientation_scenario == '180_behind':
            car_pose.orientation = airsim.to_quaternion(0, 0, np.radians(185))  # Yaw 180 degrees

        client.simSetVehiclePose(car_pose, True)
        car_controls.throttle = 1
        client.setCarControls(car_controls)
        time.sleep(1.5)
        car_controls.throttle = 0

        car_controls.steering = 0
        client.setCarControls(car_controls)
        image_response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
        image_rgb = image1d.reshape(image_response.height, image_response.width, 3)

        frame = image_rgb[44:144, 0:256, 0:3]
        # img_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)

        # Apply transformations and noise
        # transformed_img_tensor = apply_transformations(img_tensor)

        # Optionally convert back to numpy array if needed for further processing
        # augmented_frame = transformed_img_tensor.numpy().astype(np.uint8)
        # augmented_frame=apply_transformations(frame)

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
        while not done and episode_step_counter < 1000:
            min_dis = 100000
            counter_action = counter_action + 1
            episode_step_counter = episode_step_counter + 1

            total_step_counter = total_step_counter + 1

            position_info = client.getCarState()

            speed = float("{:.2f}".format(position_info.speed))
            if speed < 0:
                speed = 0
            flag_speed = check_speed(speed)

            start_time = time.time()

            # env.render()
            #   client.simPause(True)
            action = agentoo7.act(state, state2)
            #  client.simPause(False)
            steering, throttle = avail_actions_comb[action][0], avail_actions_comb[action][1]

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
            # img_tensor = tf.convert_to_tensor(new_frame, dtype=tf.float32)

            # Apply transformations and noise
            # transformed_img_tensor = apply_transformations(img_tensor)

            # Optionally convert back to numpy array if needed for further processing
            # augmented_frame = transformed_img_tensor.numpy().astype(np.uint8)
            #  augmented_frame=apply_transformations(new_frame)

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
            #     print(" reward and episodic step and done and collision",reward,episode_step_counter,done,collision_info.has_collided)

            if collision_info.has_collided == 1:
                #    print("collision")
                done = 1
                car_controls.throttle = 0
                car_controls.brake = 50
                car_controls.steering = 0
                time.sleep(1)
                client.setCarControls(car_controls)

            buffer.add((state, action, reward, next_state, int(done), state2, next_state2))

            state = next_state
            episode_total_reward += reward
            for component, value in reward_components.items():
                if component not in episode_component_sums:
                    episode_component_sums[component] = value
                else:
                    episode_component_sums[component] += value
            if total_step_counter % 5 == 0 and total_step_counter > 32:
                print("training")
                client.simPause(True)
                batch, weights, tree_idxs = buffer.sample(batch_size)
                td_error = agentoo7.train(batch, weights)
                client.simPause(False)
                print("time of train batch", time.time() - start_time)
                buffer.update_priorities(tree_idxs, td_error)
            if total_step_counter % 500 == 0:
                agentoo7.save_network()
        print("final episode reward , with epsilon probability and total steps and ip", episode_total_reward,
              agentoo7.epsilon, total_step_counter, ip_address)
        rewards_history["total_reward"].append(episode_total_reward)
        for component, sum_value in episode_component_sums.items():
            if component not in rewards_history:
                rewards_history[component] = [sum_value]
            else:
                rewards_history[component].append(sum_value)

        with open(episodic_steps_filename, 'a') as file:
            file.write(str(episode_step_counter) + '\n')
        with open(total_rewards_filename, 'a') as file:
            file.write(str(episode_total_reward) + '\n')

        #  append_rewards_as_txt(rewards_history_filename_txt, rewards_history)
        episode_step_counter = 0
        ep = ep + 1

        # Save the converted rewards history as a JSON file

        save_rewards_as_json(rewards_history_filename, rewards_history)
    print("training finished in agent,ip", ip_address, counter_model)
