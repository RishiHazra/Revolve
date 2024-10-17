import os
import random

import numpy as np
import tensorflow as tf

from rl_agent.model import DDDQN

avail_actions = np.arange(-0.8, 0.85, 0.05)
avail_throttle = np.array([0, 1])
action_grid, throttle_grid = np.meshgrid(avail_actions, avail_throttle)
combined_array = np.array([action_grid.flatten(), throttle_grid.flatten()]).T
avail_actions_comb = np.round(combined_array, 2)


class DrivingAgent:
    def __init__(self, no_model, iteration, group_id, llm_model, baseline):
        # self.no_model = no_model
        # for key, value in train_cfg.items():
        #  setattr(self, key, value)
        self.ddqn = DDDQN()
        self.q_net = self.ddqn.create_model()
        self.q_net2 = self.ddqn.create_model()
        self.iteration = iteration
        self.clipnorm = 1.5
        self.lr = 0.00025
        self.tau = 0.0075
        self.epsilon_decay = 0.001  # after 10k steps for 0.0001
        self.epsilon = 1
        self.batch_size = 32
        self.trainstep = 0
        self.min_epsilon = 0.01
        self.gamma = 0.99
        self.no_model = no_model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=self.clipnorm)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=self.clipnorm)
        self.target_net = self.ddqn.create_model()
        self.weights_dir = os.path.join(os.environ['ROOT_PATH'],
                                        f'{baseline}/{llm_model}/group_{group_id}/model_checkpoints')
        os.makedirs(self.weights_dir, exist_ok=True)

        main_weights_file = f'main1_{self.no_model}.h5'
        main_weights_file2 = f'main2_{self.no_model}.h5'
        target_weights_file = f'target_{self.no_model}.h5'

        loop_num = 1
        if loop_num != 0:
            self.epsilon = 0.5
            self.load_network()
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

    def train(self, batch, weights):
        state, action, reward, next_state, done, state2, next_state2 = batch

        states = tf.convert_to_tensor(state, dtype=tf.float32)

        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)

        dones = tf.convert_to_tensor(done, dtype=tf.float32)
        state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
        next_state2 = tf.convert_to_tensor(next_state2, dtype=tf.float32)

        gamma_tensor = tf.constant(self.gamma, dtype=tf.float32)

        actions = tf.squeeze(actions, axis=-1)
        actions = tf.cast(actions, dtype=tf.int32)
        action_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), actions], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            q_current = self.q_net([states, state2])
            q_values = tf.gather_nd(q_current, action_indices)

            q_current2 = self.q_net2([states, state2])
            q_values2 = tf.gather_nd(q_current2, action_indices)

            next_q_values_q_net = self.q_net([next_states, next_state2])
            next_q_values_q_net2 = self.q_net2([next_states, next_state2])
            next_q_values_min = tf.minimum(next_q_values_q_net, next_q_values_q_net2)

            max_action = tf.argmax(next_q_values_min, axis=1)
            max_action = tf.cast(max_action, tf.int32)

            next_q_target_net = self.target_net([next_states, next_state2])

            max_next_q_target_net = tf.gather_nd(next_q_target_net,
                                                 np.stack([np.arange(self.batch_size, dtype=np.int32), max_action],
                                                          axis=1))
            # max_next_q_target_net=tf.cast(max_next_q_target_net,dtype=tf.float16) q_target_current = tf.cast(
            # q_target_current, dtype=tf.float16)  # Ensure q_target_current is of type tf.float16

            q_target_current = rewards + gamma_tensor * max_next_q_target_net * (1 - dones)

            td_error = q_target_current - q_values
            td_error2 = q_target_current - q_values2

            weighted_squared_td_error = weights * tf.square(td_error)  # Multiply squared TD error by weights
            loss = tf.reduce_mean(weighted_squared_td_error)
            weighted_squared_td_error2 = weights * tf.square(td_error2)  # Multiply squared TD error by weights
            loss2 = tf.reduce_mean(weighted_squared_td_error2)

        gradients = tape.gradient(loss, self.q_net.trainable_variables)  # compute gradients
        self.optimizer.apply_gradients(zip(gradients, self.q_net.trainable_variables))  # apply gradients
        gradients2 = tape.gradient(loss2, self.q_net2.trainable_variables)  # compute gradients
        self.optimizer2.apply_gradients(zip(gradients2, self.q_net2.trainable_variables))
        td_error = tf.maximum(tf.abs(td_error), tf.abs(td_error2))

        self.update_epsilon()
        td_error = tf.abs(td_error).numpy()

        self.update_target()

        self.trainstep += 1
        # print("td error",td_error)
        return td_error
