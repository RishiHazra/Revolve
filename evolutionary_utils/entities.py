import glob
import json
import os
import shutil
from typing import List, Optional, Tuple

import h5py
import numpy as np


class Population:
    """A population of the rewards' database."""

    def __init__(self, reward_fn_paths: List, fitness_scores: List,
                 last_generation: Optional[List[Tuple[str, float]]]):
        self.reward_fn_paths = reward_fn_paths  # path for reward_fns
        self.fitness_scores = fitness_scores
        self.last_generation = last_generation

    @property
    def zipped(self):
        return list(zip(self.reward_fn_paths, self.fitness_scores))

    def register_reward_fn(self, reward_fn_path: str, score: float):
        self.reward_fn_paths.append(reward_fn_path)
        self.fitness_scores.append(score)

    @classmethod
    def load_population(cls, model_name: str, baseline: str):
        # loads whole population with reward fn paths and fitness scores
        # to use: Population.load_population(model_name='gpt-4')
        group_id = 0
        base_dir = os.path.join(os.environ['ROOT_PATH'],
                                f"{baseline}_database/{model_name}/group_{group_id}")
        reward_fns_file_paths = glob.glob(f"{base_dir}/reward_fns/*.txt")
        # reading the fitness score files in the same order as reward fn file paths
        fitness_scores = []
        trained_reward_fns_file_paths = []
        # cls.last_generation = []
        all_it_ids = []
        untrained_it_id = None
        for reward_fn_path in reward_fns_file_paths:
            filename = reward_fn_path.split('/')[-1]
            iteration_id, _ = filename.split('/')[-1].replace('.txt', '').split('_')
            fitness_score_filepath = f"{base_dir}/fitness_scores/{filename}"
            all_it_ids.append(iteration_id)
            try:
                fitness_scores.append(float(open(fitness_score_filepath, 'r').read()))
                trained_reward_fns_file_paths.append(reward_fn_path)
            except FileNotFoundError:
                # corner case: reward.txt exists but the policy training is underway
                untrained_it_id = iteration_id
                continue
        population_size = len(trained_reward_fns_file_paths)
        assert population_size == len(fitness_scores), \
            "list of Fitness Scores should be the same size as the list of Reward Functions"

        # dropping the current generation (which is still training)
        if untrained_it_id is not None:
            all_it_ids = [id for id in all_it_ids if id not in untrained_it_id]
        last_it_id = max(all_it_ids)  # last generation iteration id

        def is_last_gen(path, gen_id):
            if gen_id == path.split('/')[-1].replace('.txt', '').split('_')[0]:
                return True
            return False

        last_generation = [(rew_fn_path, score) for rew_fn_path, score in
                           zip(reward_fns_file_paths, fitness_scores) if is_last_gen(rew_fn_path, last_it_id)]
        return cls(trained_reward_fns_file_paths, fitness_scores, last_generation)

    @staticmethod
    def load_reward_history(model_name: str, baseline: str) -> List[str]:
        # loads all reward history for model training with llm_model: gpt-4
        # to use: Population.load_reward_history(model_name='gpt-4')
        group_id = 0
        base_dir = os.path.join(os.environ['ROOT_PATH'],
                                f"{baseline}_database/{model_name}/group_{group_id}")
        reward_history_paths = glob.glob(f"{base_dir}/reward_history/*.json")
        reward_history = [json.load(open(filename, 'r')) for filename in reward_history_paths]
        return reward_history

    @staticmethod
    def load_reward_fns(model_name: str, baseline: str) -> List[str]:
        # loads all reward fns for model_name: gpt-4
        # to use: Population.load_reward_fns(model_name='gpt-4')
        group_id = 0
        base_dir = os.path.join(os.environ['ROOT_PATH'],
                                f"{baseline}_database/{model_name}/group_{group_id}")
        reward_files = glob.glob(f"{base_dir}/reward_fns/*.txt")
        reward_fns = [open(filename, 'r').read() for filename in reward_files]
        return reward_fns

    @staticmethod
    def load_model_checkpoints(model_name: str, baseline: str):
        # loads all trained model checkpoints for model_name: gpt-4
        # to use: Population.load_trained_policies(model_name='gpt-4')
        group_id = 0
        base_dir = os.path.join(os.environ['ROOT_PATH'],
                                f"{baseline}_database/{model_name}/group_{group_id}")
        model_checkpoint_paths = glob.glob(f"{base_dir}/model_checkpoints/*.h5")
        models = [h5py.File(filename, 'r') for filename in model_checkpoint_paths]
        return models


class Group:
    """A subpopulation of the rewards' database."""

    def __init__(self, reward_fn_paths: List, fitness_scores: List, size: int):
        self.reward_fn_paths = reward_fn_paths  # path for reward_fns
        self.fitness_scores = fitness_scores  # uses human fitness scores for human feedback
        self.size = size

    @property
    def zipped(self):
        return list(zip(self.reward_fn_paths, self.fitness_scores))

    def remove_lowest(self):
        lowest_score_index = np.argmin(self.fitness_scores)
        Group.remove_files(self.reward_fn_paths[lowest_score_index])
        del self.reward_fn_paths[lowest_score_index]
        # del self.arguments[lowest_score_index]
        del self.fitness_scores[lowest_score_index]
        self.size -= 1

    def register_reward_fn(self, reward_fn_path: str, score: float):
        self.reward_fn_paths.append(reward_fn_path)
        # self.arguments.append(arguments)
        self.fitness_scores.append(score)
        self.size += 1

    def migrate_reward_fn(self, founder_reward_fn_path: str, founder_score: float,
                          founder_group_id: int, reset_group_id: int):
        base_dir = '/'.join(founder_reward_fn_path.split('/')[:-3])  # root_path/database/gpt-4
        # Parse out the iteration id and counter
        iteration, counter = founder_reward_fn_path.split('/')[-1].replace('.txt', '').split('_')
        reset_reward_fn_path = f'{base_dir}/group_{reset_group_id}/reward_fns/{iteration}_{counter}.txt'
        self.register_reward_fn(reset_reward_fn_path, founder_score)
        Group.copy_files(iteration, counter, founder_group_id, reset_group_id, base_dir)

    @classmethod
    def load_group(cls, group_id: int, model_name: str, baseline: str):
        # loads all group members with reward fn paths and fitness scores
        # to use: Group.load_group(group_id=*, model_name='gpt-4')
        # fitness_type: 'auto' for auto feedback fitness scores, '' otherwise
        base_dir = os.path.join(os.environ['ROOT_PATH'],
                                f"{baseline}_database/{model_name}/group_{group_id}")
        reward_fns_file_paths = glob.glob(f"{base_dir}/reward_fns/*.txt")
        # reading the fitness score files in the same order as reward fn file paths
        fitness_scores = []
        trained_reward_fns_file_paths = []
        for reward_fn_path in reward_fns_file_paths:
            filename = reward_fn_path.split('/')[-1]
            fitness_type = f'_auto' if 'auto' in baseline else ''
            fitness_score_filepath = f"{base_dir}/fitness_scores{fitness_type}/{filename}"
            try:
                fitness_scores.append(float(open(fitness_score_filepath, 'r').read()))
                trained_reward_fns_file_paths.append(reward_fn_path)
            except FileNotFoundError:
                # corner case: reward.txt exists but the policy training is underway
                continue
        group_size = len(trained_reward_fns_file_paths)
        # fitness_scores_filepaths = glob.glob(f"{base_dir}/fitness_scores/*.txt")
        # fitness_scores = [float(open(filename, 'r').read())
        #                   for filename in fitness_scores_filepaths]
        assert group_size == len(fitness_scores), \
            "list of Fitness Scores should be the same size as the list of Reward Functions"
        return cls(trained_reward_fns_file_paths, fitness_scores, group_size)

    @staticmethod
    def load_reward_history(group_id: int, model_name: str, baseline: str) -> List[str]:
        # loads all reward history for model training in group_id and llm_model: gpt-4
        # to use: Group.load_reward_history(group_id=*, model_name='gpt-4')
        database_name = f'{baseline}_database'
        base_dir = os.path.join(os.environ['ROOT_PATH'],
                                f"{database_name}/{model_name}/group_{group_id}")
        reward_history_paths = glob.glob(f"{base_dir}/reward_history/*.json")
        reward_history = [json.load(open(filename, 'r')) for filename in reward_history_paths]
        return reward_history

    @staticmethod
    def load_reward_fns(group_id: int, model_name: str, baseline: str) -> List[str]:
        # loads all reward fns for group_id and model_name: gpt-4
        # to use: Group.load_reward_fns(group_id=*, model_name='gpt-4')
        database_name = f'{baseline}_database'
        base_dir = os.path.join(os.environ['ROOT_PATH'],
                                f"{database_name}/{model_name}/group_{group_id}")
        reward_files = glob.glob(f"{base_dir}/reward_fns/*.txt")
        reward_fns = [open(filename, 'r').read() for filename in reward_files]
        return reward_fns

    @staticmethod
    def load_model_checkpoints(group_id: int, model_name: str, baseline: str):
        # loads all trained model checkpoints for group_id and model_name: gpt-4
        # to use: Group.load_trained_policies(group_id=*, model_name='gpt-4')
        database_name = f'{baseline}_database'
        base_dir = os.path.join(os.environ['ROOT_PATH'],
                                f"{database_name}/{model_name}/group_{group_id}")
        model_checkpoint_paths = glob.glob(f"{base_dir}/model_checkpoints/*.h5")
        models = [h5py.File(filename, 'r') for filename in model_checkpoint_paths]
        return models

    @staticmethod
    def load_human_feedback(group_id: int, model_name: str) -> List[str]:
        # loads NL Human Feedback for group_id and model_name: gpt-4
        # to use: Group.load_human_feedback(group_id=*, model_name='gpt-4')
        base_dir = os.path.join(os.environ['ROOT_PATH'],
                                f"revolve_database/{model_name}/group_{group_id}")
        human_feedback_files = glob.glob(f"{base_dir}/human_feedback/*.txt")
        human_feedbacks = [open(filename, 'r').read() for filename in human_feedback_files]
        return human_feedbacks

    @staticmethod
    def remove_files(filename: str):
        # delete files from the database for a certain iteration and model_count
        # deletes reward_history, model checkpoint, reward_fn txt file
        # filename (reward fn): root_path/database/{model_name}/group_{group_id}/reward_fns/it_model.txt
        def delete_file(filepath: str, filetype: str):
            if os.path.exists(filepath):
                print(f'\nRemoving {filetype} From {filepath}.\n')
                os.remove(filepath)
            else:
                print(f'\n{filetype} Does Not Exist In {filepath}.\n')

        base_dir = '/'.join(filename.split('/')[:-2])  # root_path/database/gpt-4/group_*/
        # Parse out the iteration id and model id
        iteration_id, model_id = filename.split('/')[-1].replace('.txt', '').split('_')
        reward_fn_file = filename
        reward_history_file = f"{base_dir}/reward_history/{iteration_id}_{model_id}.json"
        fitness_score_auto_filename = f"{base_dir}/fitness_scores_auto/{iteration_id}_{model_id}.txt"
        fitness_score_filename = f"{base_dir}/fitness_scores/{iteration_id}_{model_id}.txt"
        human_feedback_filename = f"{base_dir}/human_feedback/{iteration_id}_{model_id}.txt"
        # model checkpoints split into 3
        model_filenames = glob.glob(f"{base_dir}/model_checkpoints/*_{iteration_id}_{model_id}.h5")

        delete_file(reward_fn_file, f'reward fn (.txt) file')
        delete_file(reward_history_file, 'reward history (.json) file')
        for model_filename in model_filenames:
            delete_file(model_filename, 'model checkpoint (.h5) file')
        delete_file(fitness_score_auto_filename, 'fitness score auto (.txt) file')
        delete_file(fitness_score_filename, 'fitness score (.txt) file')
        delete_file(human_feedback_filename, 'human feedback (.txt) file')

    @staticmethod
    def copy_files(iteration: int, counter: int, from_group: int, to_group: int, base_dir: str):
        def copy_file(dir_type: str):
            dir2file_type = {'reward_history': 'json', 'model_checkpoints': 'h5',
                             'fitness_scores_auto': 'txt', 'reward_fns': 'txt',
                             'fitness_scores': 'txt', 'human_feedback': 'txt'}
            filetype = dir2file_type[dir_type]
            dst_folder = f'{base_dir}/group_{to_group}/{dir_type}'
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            src_files = glob.glob(f'{base_dir}/group_{from_group}/{dir_type}/*{iteration}_{counter}.{filetype}')
            for src_file in src_files:
                assert os.path.exists(src_file), 'Path for founder model does not exist.'
                shutil.copy(src_file, dst_folder)

        copy_file('reward_history')
        copy_file('fitness_scores_auto')
        copy_file('reward_fns')
        copy_file('model_checkpoints')
        if 'auto' not in base_dir:
            copy_file('fitness_scores')
            copy_file('human_feedback')
