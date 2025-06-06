"""
Various stages of individual generation, training, and evaluation:
1. Reward Function Generation
2. Policy Training
3. Policy Evaluation
"""

import concurrent.futures
import json
import multiprocessing
import os
import time
from typing import Tuple, Optional, Dict

import absl.logging as logging
import hydra
import openai
from hydra.core.global_hydra import GlobalHydra
from openai import OpenAI

from rl_agent.evaluate import return_score
# from rl_agent.generate_scores import generate_behaviour
from rl_agent.main import run_training
from utils import parse_llm_output, serialize_dict, format_human_feedback

openai_api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


# generates reward functions
class RewardFunctionGeneration:
    def __init__(self, system_prompt: str, env_input: str):
        # TODO: change system message based on Eureka
        self.system_prompt = system_prompt
        self.env_input = env_input  # env_class + task
        self.llm = "gpt-4-1106-preview"

    def query_llm(self, in_context_prompt: str) -> Tuple[str, int, int]:
        response = client.chat.completions.create(
            model=self.llm,  # gpt-4-1106-preview, gpt-3.5-turbo-1106
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt + "\n" + self.env_input,
                },
                {"role": "user", "content": f"{in_context_prompt}"},
            ],
            temperature=1,
            max_tokens=4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    @staticmethod
    def prepare_in_context_prompt(
            in_context_samples: Optional[List[Tuple[str, float]]],
            operator_prompt: str,
            evolve: bool,
            baseline: str,
    ) -> str:
        # prepares a prompt from in context examples sampled from RewardsDatabase
        in_context_samples_str = ""
        if not evolve:
            return in_context_samples_str
        for filename, fitness_score in in_context_samples:
            in_context_samples_str += "\n\n```python\n"
            in_context_samples_str += open(filename, "r").read()
            in_context_samples_str += "\n```\n"
            reward_history_file = filename.replace(
                "generated_fns", "reward_history"
            ).replace(".txt", ".json")
            # reward_history = json.load(open(reward_history_file, "r"))

            reward_history = []
            with open(reward_history_file, "r") as f:
                for line in f:
                    reward_history.append(json.loads(line.strip()))

            combined_components = {}
            for entry in reward_history:
                for key, value in entry["episode_components"].items():
                    if key not in combined_components:
                        combined_components[key] = []
                    combined_components[key].append(value)
            in_context_samples_str += f"fitness score: {fitness_score}"
            in_context_samples_str += f"\n{serialize_dict(combined_components)}"
            if "auto" not in baseline:
                # human feedback
                human_feedback_file = filename.replace(
                    "generated_fns", "human_feedback"
                )
                human_feedback = open(human_feedback_file, "r").read()
                human_feedback = format_human_feedback(human_feedback)
                in_context_samples_str += f"\nhuman feedback: {human_feedback}"
        operator_prompt = operator_prompt.replace(
            "\n\n<EXAMPLES>", in_context_samples_str
        )
        operator_prompt = operator_prompt.replace("<EPISODES>", "100")
        return operator_prompt

    def generate_rf(self, in_context_prompt: str) -> str:
        parsed_function_str = None
        while True:
            try:
                raw_llm_output, _, _ = self.query_llm(in_context_prompt)
                parsed_function_str = parse_llm_output(raw_llm_output)
                break
            # except openai.RateLimitError or openai.APIError or openai.Timeout:
            except openai.RateLimitError or openai.APIError or openai.Timeout:
                time.sleep(10)
                continue
        # parsed_function_str = open("test_heuristic", "r").read()
        return parsed_function_str


class TrainPolicy:
    """
    Train RL Policy
    """

    def __init__(
            self,
            reward_fn_str: str,
            generation_id: int,
            counter_id: int,
            island_id: int,
            baseline: str,
            output_log: str,
    ):
        self.train_cfg = None
        self._load_train_cfg()

        self.reward_func_str = reward_fn_str
        self.island_id = island_id
        self.generation_id = generation_id
        self.counter_id = counter_id
        self.baseline = baseline  # ['revolve', 'revolve_auto', 'eureka', 'eureka_auto']
        self.output_log = output_log
        logging.info(
            f"Initializing TrainPolicy: generation_id={generation_id}, island_id={island_id}, type(island_id)={type(island_id)}"
        )

    def _load_train_cfg(self):
        logging.info("Loading train cfg")

        # Ensure ROOT_PATH exists
        root_path = os.environ.get("ROOT_PATH")
        if not root_path:
            raise EnvironmentError("ROOT_PATH environment variable is not set.")

        # Convert absolute path to relative
        config_relative_path = os.path.relpath(
            os.path.join(root_path, "cfg"), start=os.getcwd()
        )

        # Clear Hydra global state if already initialized
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Initialize Hydra with the relative config path
        with hydra.initialize(config_path=config_relative_path):
            self.train_cfg = hydra.compose(config_name="train")
            logging.info("Training Config loaded")

    def train_policy(self) -> Tuple[str, str]:
        # This will define the compute_reward function dynamically

        reward_history_file = os.path.join(
            self.output_log,
            f"island_{self.island_id}/reward_history/{self.generation_id}_{self.counter_id}.json",
        )

        checkpoint_file = os.path.join(
            self.output_log,
            f"island_{self.island_id}/model_checkpoints/{self.generation_id}_{self.counter_id}.h5",
        )

        velocity_file_path = os.path.join(
            self.output_log,
            f"island_{self.island_id}/velocity_logs/velocity_{self.generation_id}_{self.counter_id}.txt",
        )

        model_checkpoint_path = os.path.join(
            self.output_log, f"island_{self.island_id}/model_checkpoints"
        )

        fitness_file = os.path.join(
            self.output_log,
            f"island_{self.island_id}/fitness_scores/{self.generation_id}_{self.counter_id}.txt",
        )

        log_dir = os.path.join(
            self.output_log,
            f"island_{self.island_id}/log_dir/{self.generation_id}_{self.counter_id}",
        )

        run_training(
            self.reward_func_str,
            self.island_id,
            self.generation_id,
            self.counter_id,
            reward_history_file,
            model_checkpoint_path,
            fitness_file,
            velocity_file_path,
            self.output_log,
            log_dir,
        )
        return checkpoint_file, velocity_file_path


# human evaluation, fitness functions
class RewardFunctionEvaluation:
    """
    Fitness Function Evaluator
    """

    def __init__(self, baseline: str):
        self.baseline = baseline

    # def generate_behavior(self, filename: str) -> Dict:
    #     # be provided
    #     reward_history_dict = json.load(open(filename, "r"))
    #     return reward_history_dict

    @staticmethod
    def evaluate_behavior(
            full_velocity_log_path,
    ) -> Dict[str, float]:
        fitness_score = return_score(full_velocity_log_path)
        return {"fitness": fitness_score}


def train_policies_in_parallel(
        policy_classes: List[TrainPolicy],
) -> List[Tuple[str, str]]:
    """
    submit multiple training policies in parallel
    """
    multiprocessing.set_start_method("spawn", force=True)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(policy_classes)
    ) as executor:
        futures = [
            executor.submit(policy_class.train_policy)
            for policy_class in policy_classes
        ]
        #  results = executor.map(train_model, enumerate(model_classes))

        results = [future.result() for future in futures]
    return results


def evaluate_policies_in_parallel(
        ckpt_and_performance_paths: List[Tuple[str, str]]
) -> List[Dict[str, float]]:
    """
    Submit evaluation tasks in parallel with both checkpoint paths and velocity paths.
    """
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(ckpt_and_performance_paths)
    ) as executor:
        futures = [
            executor.submit(RewardFunctionEvaluation.evaluate_behavior, velocity_path)
            for _, velocity_path in ckpt_and_performance_paths
        ]

        fitness_dicts = [future.result() for future in futures]
    return fitness_dicts
