"""
Evolutionary Utilities
"""

import glob
import json
import os
import sys
from typing import List, Optional

import numpy as np
from absl import logging


class Individual:
    """Single Individual in an Island"""

    def __init__(
        self,
        island_id: int,
        generation_id: int,
        counter_id: int,
        rew_fn_string: str,
        fitness_score: float,
        metrics_dict: dict,  # other performance metrics
        reward_fn_dir: str,
    ):
        self.island_id = island_id
        self.generation_id = generation_id
        self.counter_id = counter_id
        self.rew_fn_string = rew_fn_string
        self.fitness_score = fitness_score
        self.reward_history = None
        self.model_checkpoint = None
        self.metrics_dict = metrics_dict
        self.reward_fn_dir = reward_fn_dir

    @property
    def fn_file_path(self):
        """function string path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/generated_fns/"
            f"{self.generation_id}_{self.counter_id}.txt"
        )

    @property
    def fitness_file_path(self):
        """fitness score file path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/fitness_scores/"
            f"{self.generation_id}_{self.counter_id}.txt"
        )

    @property
    def reward_history_path(self):
        """reward history file path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/reward_history/"
            f"{self.generation_id}_{self.counter_id}.json"
        )

    @property
    def model_checkpoint_path(self):
        """model (policy) checkpoint path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/model_checkpoints/"
            f"{self.generation_id}_{self.counter_id}.h5"
        )

    @property
    def human_feedback_path(self):
        """human feedback path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/human_feedback/"
            f"{self.generation_id}_{self.counter_id}.h5"
        )

    def save_files(self):
        """
        saves fitness score and fn string
        """
        base_path = f"{self.reward_fn_dir}/island_{self.island_id}"
        for dir_name in [
            "generated_fns",
            "fitness_scores",
            "reward_history",
            "model_checkpoints",
        ]:
            dir_path = os.path.join(base_path, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # save fn string
        with open(self.fn_file_path, "w") as outfile:
            outfile.write(self.rew_fn_string)
        # save fitness score
        with open(self.fitness_file_path, "w") as outfile:
            serialized_dict = json.dumps(self.metrics_dict)
            outfile.write(serialized_dict)
        # save reward history
        # TODO: reward history to be dumped from the RL training code?
        # save model_checkpoint
        # TODO: model checkpoint to be dumped from the RL training code?

    def remove_files(self):
        """
        delete associated files from the database for the individual
        deletes generated_fn txt file, fitness score, reward history, model checkpoint file
        fn_path: root_path/database/island_{island_id}/generated_fns/gen_counter.txt
        """

        def delete_file(filepath: str, filetype: str):
            if os.path.exists(filepath):
                logging.info(f"Removing {filetype} from {filepath}.")
                os.remove(filepath)
            else:
                logging.info(f"{filetype} does not exist in {filepath}.")

        delete_file(self.fn_file_path, "generated reward fn (.txt) file")
        delete_file(self.fitness_file_path, "fitness score (.txt) file")
        delete_file(self.reward_history_path, "reward history (.json) file")
        delete_file(self.model_checkpoint_path, "model checkpoint (.h5) file")


class Island:
    """A population of individuals: aka island"""

    def __init__(
        self,
        island_id: int,
        generation_ids: List[int],
        counter_ids: List[int],
        rew_fn_strings: List[str],
        fitness_scores: List[float],
        metrics_dicts: List[dict],
        reward_fn_dir: str,
        baseline: str,
    ):
        self.reward_fn_dir = reward_fn_dir
        self.island_id = island_id
        self.individuals = [
            Individual(
                self.island_id,
                generation_id,
                counter_id,
                rew_fn_str,
                fitness_score,
                metrics_dict,
                self.reward_fn_dir,
            )
            for generation_id, counter_id, rew_fn_str, fitness_score, metrics_dict in zip(
                generation_ids,
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                metrics_dicts,
            )
        ]
        self.baseline = baseline

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def fitness_scores(self) -> List[float]:
        if self.size == 0:
            return [-sys.maxsize - 1]
        return [individual.fitness_score for individual in self.individuals]

    @property
    def best_fitness_score(self) -> float:
        return max(self.fitness_scores)

    @property
    def average_fitness_score(self):
        return np.mean(self.fitness_scores)

    @property
    def fittest_individual(self) -> Individual:
        fittest_ind = np.argmax(
            [individual.fitness_score for individual in self.individuals]
        )
        return self.individuals[fittest_ind]

    @property
    def generation_ids(self) -> List[int]:
        return [individual.generation_id for individual in self.individuals]

    @property
    def counter_ids(self) -> List[int]:
        return [individual.counter_id for individual in self.individuals]

    @property
    def fn_file_paths(self) -> List[str]:
        return [individual.fn_file_path for individual in self.individuals]

    @property
    def reward_history_paths(self) -> List[str]:
        return [individual.reward_history_path for individual in self.individuals]

    @property
    def human_feedback_paths(self) -> Optional[List[str]]:
        if "auto" not in self.baseline:
            return None
        return [individual.reward_history_path for individual in self.individuals]

    def register_individual_in_island(
        self,
        generation_id: int,
        counter_id: int,
        rew_fn_string: str,
        fitness_score: float,
        metrics_dict: dict,
    ):
        """
        add individual to the island population.
        """

        logging.info(
            f"Registering Individual with generation_id {generation_id}"
            f" | counter_id {counter_id} in Island {self.island_id}."
        )
        new_individual = Individual(
            self.island_id,
            generation_id,
            counter_id,
            rew_fn_string,
            fitness_score,
            metrics_dict,
            self.reward_fn_dir,
        )
        self.individuals.append(new_individual)
        new_individual.save_files()

    @classmethod
    def load_island(cls, heuristic_dir: str, baseline: str, island_id: int = 0):
        """
        utility function
        loads island population with fn paths and fitness scores
        to use: Population.load_population(island_id=*)
        island_id = 0 for a single island (default)
        """

        base_dir = os.path.join(f"{heuristic_dir}/island_{island_id}")
        all_fns_file_paths = glob.glob(f"{base_dir}/generated_fns/*.txt")
        # reading the fitness score files in the same order as generated fn file paths
        fitness_scores = []
        all_generation_ids = []
        all_counter_ids = []
        all_fn_strings = []
        all_metrics = []

        for fn_path in all_fns_file_paths:
            filename = fn_path.split("/")[-1]
            generation_id, counter_id = (
                filename.split("/")[-1].replace(".txt", "").split("_")
            )
            fitness_score_filepath = f"{base_dir}/fitness_scores/{filename}"
            # load performance metrics and deserialize the dictionary
            with open(fitness_score_filepath, "r") as f:
                metrics_dict = json.loads(f.read().strip("'").strip("\\"))
            all_metrics.append(metrics_dict)
            # use the desired metric as the fitness score
            fitness_score = metrics_dict["fitness"]
            fitness_scores.append(float(fitness_score))
            all_generation_ids.append(int(generation_id))
            all_counter_ids.append(int(counter_id))
            fn_string = open(fn_path, "r").read()
            all_fn_strings.append(fn_string)

        return cls(
            island_id,
            all_generation_ids,
            all_counter_ids,
            all_fn_strings,
            fitness_scores,
            all_metrics,
            heuristic_dir,
            baseline,
        )

    def remove_lowest(self):
        """
        removes the individual with the lowest fitness score in the island
        """

        lowest_score_index = np.argmin(
            [individual.fitness_score for individual in self.individuals]
        )
        weakest_individual = self.individuals.pop(lowest_score_index)
        weakest_individual.remove_files()

    def remove_individual(self, to_remove_individual: Individual):
        """
        remove an individual from the island
        """
        for individual_id, individual in enumerate(self.individuals):
            if (
                individual.generation_id == to_remove_individual.generation_id
                and individual.counter_id == to_remove_individual.counter_id
            ):
                self.individuals.pop(individual_id)
                individual.remove_files()
                return

    def only_keep_best(self):
        """
        remove all except best individual
        """
        fittest_individual = self.fittest_individual
        # TODO: handle case where more than one are equally fit
        for individual in self.individuals:
            if individual.fitness_score == fittest_individual.fitness_score:
                continue
            individual.remove_files()
        self.individuals = [fittest_individual]

    def migrate_fn(
        self,
        founder_individual: Individual,
    ):
        """
        migrate individual from founder island to current island
        """

        # register founder fn in the reset island
        self.register_individual_in_island(
            founder_individual.generation_id,
            founder_individual.counter_id,
            founder_individual.rew_fn_string,
            founder_individual.fitness_score,
            founder_individual.metrics_dict,
        )
