import random
import sys
from operator import itemgetter
from typing import Tuple, List, Dict

import numpy as np
from absl import logging

from evolutionary_utils.entities import Group, Population


def normalized(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class RewardsDatabase:
    """
    Adapted from Fun Search: https://github.com/google-deepmind/funsearch/blob/main
    """

    def __init__(self, num_groups: int, max_size: int, crossover_prob: float,
                 model_name: str, load_groups: bool, baseline: str):
        self.num_groups = num_groups  # starting with num_groups, does not increase with crossover
        self.max_size = max_size  # max group size
        self.crossover_prob = crossover_prob
        self.model_name = model_name

        if load_groups:
            # for it > 0, load stored groups
            groups = []
            best_scores = []
            # actual_num_groups = len(os.listdir(os.path.join(os.environ['ROOT_PATH'],
            #                                                 f"database/{model_name}")))
            # for group_id in range(actual_num_groups):
            for group_id in range(self.num_groups):
                groups.append(Group.load_group(group_id, model_name, baseline))
                if len(groups[-1].reward_fn_paths) == 0:
                    best_scores.append(-sys.maxsize - 1)
                else:
                    best_scores.append(max(groups[-1].fitness_scores))
            self._groups = groups
            self._best_score_per_group = best_scores
            # self._best_reward_per_group = [None] * self.num_groups  # stores best reward fn in each group
        else:
            # Initialize empty islands.
            self._groups = [Group([], [], 0) for _ in range(self.num_groups)]
            self._best_score_per_group: List[float] = [-sys.maxsize - 1] * self.num_groups

    def add_reward_to_group(self, reward_fn_path: List[str],
                            fitness_scores: List[float], group_ids: List[int]):
        for rew_fn_path, fitness_score, group_id in zip(reward_fn_path, fitness_scores, group_ids):
            # corner case: if group is not empty, calculate average fitness score
            if self._groups[group_id].size != 0:
                group_avg_fitness_score = np.array(self._groups[group_id].fitness_scores).mean()
            else:
                group_avg_fitness_score = -sys.maxsize - 1
            # check if reward is adding any value to the group
            if fitness_score > group_avg_fitness_score:
                self._groups[group_id].register_reward_fn(rew_fn_path, fitness_score)
                # self._best_reward_per_group[group_id] = rew_fn_path
                # self._best_scores_per_test_per_group[group_id] = scores_per_test
                self._best_score_per_group[group_id] = fitness_score
                logging.info('Average score of group %d increased to %s', group_id,
                             np.array(self._groups[group_id].fitness_scores).mean())
            else:
                # delete the stored rewards txt, models, json
                logging.info('Fitness score %s for group %d lower than average group reward %s, discarding',
                             fitness_score, group_id, group_avg_fitness_score)
                Group.remove_files(rew_fn_path)

            # if group size exceeds max size, discard reward with the lowest score
            if self._groups[group_id].size > self.max_size:
                logging.info('Exceeded maximum size in group %d, '
                             'discarding reward with lowest score', group_id)
                while self._groups[group_id].size > self.max_size:
                    self._groups[group_id].remove_lowest()

        # happens at the end of each iteration
        # reset_prob = (len(self._groups) - self.num_groups) / self.num_groups
        if random.random() > 0.8:
            self.reset_groups()

    def reset_groups(self):
        """Resets the weaker half of groups."""
        # sort best scores after adding minor noise to break ties.
        indices_sorted_by_score = np.argsort(self._best_score_per_group +
                                             np.random.randn(len(self._best_score_per_group)) * 1e-6)
        num_groups_to_reset = len(self._groups) // 2
        reset_groups_ids = indices_sorted_by_score[:num_groups_to_reset]
        keep_groups_ids = indices_sorted_by_score[num_groups_to_reset:]
        for group_id in reset_groups_ids:
            # discard the group members
            group_rew_filenames = self._groups[group_id].reward_fn_paths
            self._groups[group_id] = Group([], [], 0)
            self._best_score_per_group[group_id] = -sys.maxsize - 1
            # delete associated files
            for rew_filename in group_rew_filenames:
                Group.remove_files(rew_filename)
            # founder group to initialize/seed the empty group with
            founder_group_id = np.random.choice(keep_groups_ids)
            # the best member from the founder group is used to seed the reset group
            founder_rew_id = np.argmax(self._groups[founder_group_id].fitness_scores)
            founder_rew_fn_path = self._groups[founder_group_id].reward_fn_paths[founder_rew_id]
            founder_fitness_score = self._groups[founder_group_id].fitness_scores[founder_rew_id]
            # register the new (seed) member of the reset group and
            # copy/migrate the relevant files from founder group to the reset group
            self._groups[group_id].migrate_reward_fn(founder_rew_fn_path, founder_fitness_score,
                                                     founder_group_id, group_id)

    def sample_in_context(self, num_samples: Dict) -> Tuple[List[Tuple[str, float]], int, str]:
        # returns a tuple of sampled reward_fns and its corresponding group
        # for crossover: create new group, for mutation: return sampled group id
        operator = None
        # selecting the groups to mutate/crossover based on average fitness score
        # this ensures that the groups explore + exploit
        average_fitness_scores = normalized([np.array(self._groups[group_id].fitness_scores).mean()
                                             for group_id in range(self.num_groups)])

        if random.random() >= self.crossover_prob:
            # making mutation more likely leading to utilizing current groups
            # first sample a group
            print(f'\nMutation\n: num groups: {len(self._groups)}')
            sampled_group_id, sampled_group = random.choices(list(enumerate(self._groups)),
                                                             weights=average_fitness_scores)[0]
            # then sample without replacement num_samples reward fns
            in_context_sample_ids = np.random.choice(range(len(sampled_group.reward_fn_paths)),
                                                     p=normalized(sampled_group.fitness_scores),
                                                     size=num_samples['mutation'], replace=False)
            in_context_samples = list(zip(np.array(sampled_group.reward_fn_paths)[in_context_sample_ids],
                                          np.array(sampled_group.fitness_scores)[in_context_sample_ids]))
            operator = 'mutation'
        else:
            # sample num_samples groups without replacement
            print(f'\nCrossover\n: num groups: {len(self._groups)}')
            # TODO: handle sampling of empty groups
            sampled_group_ids = np.random.choice(range(len(self._groups)), replace=False,
                                                 p=average_fitness_scores,
                                                 size=num_samples['crossover'])
            sampled_groups_avg_scores = average_fitness_scores[sampled_group_ids]
            in_context_samples = []
            sample_scores = []
            for ind in sampled_group_ids:
                sample = random.choices(self._groups[ind].zipped,
                                        weights=normalized(self._groups[ind].fitness_scores))[0]
                in_context_samples.append(sample)
                sample_scores.append(sample[1])
            # then sample a reward fn str from each group
            # add the new crossed reward fn to the group with a lower average fitness score
            from_group_id, to_group_id = sampled_group_ids[np.argmax(sampled_groups_avg_scores)], \
                sampled_group_ids[np.argmin(sampled_groups_avg_scores)]
            sampled_group_id = to_group_id

            # if migrate from_group_id to to_group_id
            # 1. copy the rew fn to the new group; 2. delete from the from_group_id
            # from_rew_fn_path = in_context_samples[0][0]
            # self._groups[to_group_id].migrate_reward_fn(from_rew_fn_path,
            #                                             sample_scores[np.argmax(sampled_groups_avg_scores)],
            #                                             from_group_id, to_group_id)
            # Group.remove_files(from_rew_fn_path)

            # self._groups.append(Group([], [], 0))
            # self._best_score_per_group.append(-sys.maxsize - 1)
            # self._best_reward_per_group.append(None)
            # self.num_groups += 1
            operator = 'crossover'
        # each sample in 'in_context_samples' is a tuple of (reward_fn_path: str, fitness_score: float)
        return in_context_samples, sampled_group_id, operator


class EurekaDatabase:
    def __init__(self, model_name: str, load_population: bool, baseline: str):
        # self.population_size = population_size
        self.model_name = model_name

        if load_population:
            # for it > 0, load stored population
            self._population = Population.load_population(model_name, baseline)
        else:
            # Initialize empty population
            self._population = Population([], [], None)

    def update_rewards(self, reward_fn_path: List[str], fitness_scores: List[float]):
        for rew_fn_path, fitness_score in zip(reward_fn_path, fitness_scores):
            self._population.register_reward_fn(rew_fn_path, fitness_score)

    def sample_in_context(self) -> List[Tuple[str, float]]:
        # TODO: For in-context learning, eureka picks the best sample from the last generation
        in_context_sample = []
        last_gen = self._population.last_generation
        in_context_sample.append(max(last_gen, key=itemgetter(1)))
        return in_context_sample
