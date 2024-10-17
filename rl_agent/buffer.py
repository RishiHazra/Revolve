import random

import numpy as np
import tensorflow as tf


class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size=30000, eps=1e-2, alpha=0.9, beta=0.4):
        self.tree = SumTree(size=buffer_size)
        self.size = buffer_size

        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        self.buffer_size = buffer_size
        # transition: state, action, reward, next_state, done
        self.state = np.zeros((self.buffer_size, 100, 256, 12), dtype=np.float32)
        self.action = np.zeros((self.buffer_size, 1), dtype=np.int32)
        self.reward = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_state = np.zeros((self.buffer_size, 100, 256, 12), dtype=np.float32)
        self.done = np.zeros(buffer_size, dtype=np.int32)
        self.state2 = np.zeros((self.buffer_size, 6), dtype=np.float32)
        self.next_state2 = np.zeros((self.buffer_size, 6), dtype=np.float32)
        self.count = 0
        self.real_size = 0

    def add(self, transition):
        state, action, reward, next_state, done, state2, next_state2 = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state[self.count] = tf.convert_to_tensor(state)
        self.action[self.count] = tf.convert_to_tensor(action)
        self.reward[self.count] = tf.convert_to_tensor(reward)
        self.next_state[self.count] = tf.convert_to_tensor(next_state)
        self.done[self.count] = tf.convert_to_tensor(done)
        self.state2[self.count] = tf.convert_to_tensor(state2)
        self.next_state2[self.count] = tf.convert_to_tensor(next_state2)
        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):

        sample_idxs, tree_idxs = [], []
        priorities = np.empty((batch_size, 1), dtype=np.float16)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i.
        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** -self.beta

        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs],
            self.action[sample_idxs],
            self.reward[sample_idxs],
            self.next_state[sample_idxs],
            self.done[sample_idxs],
            self.state2[sample_idxs],
            self.next_state2[sample_idxs]
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
