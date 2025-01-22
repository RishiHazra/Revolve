import inspect
import os
import random
import math
import fcntl
from copy import copy, deepcopy
from collections import Counter
from typing import Optional, Callable, List, Tuple, Dict
import torch

import numpy as np


class DataLogger:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path

    def log(self, data_sample):
        with open(self.log_file_path, "a", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write(str(data_sample) + "\n")
            fcntl.flock(file, fcntl.LOCK_UN)


def define_function_from_string(
    function_string: str,
) -> Tuple[Optional[Callable], List[str]]:
    """
    Takes a string containing a function definition and returns the defined function.

    Args:
    - function_string (str): The string containing the function definition.

    Returns:
    - function: The defined function.
    """
    namespace = {}
    # TODO: add more additional globals?

    # TODO: add more additional globals?
    additional_globals = {
        "math": math,
        "torch": torch,
        "np": np,
        "Tuple": Tuple,
        "List": List,
        "Callable": Callable,
        "Optional": Optional,
        "Dict": Dict,
        "copy": copy,
        "deepcopy": deepcopy,
        "random": random,
    }
    namespace.update(additional_globals)
    exec(function_string, namespace)
    # TODO: change 'compute_reward' to some other identifier
    function = next(
        (value for key, value in namespace.items() if key == "compute_reward"), None
    )
    args = inspect.getfullargspec(function).args if function else []
    return function, args


def fix_indentation(code, spaces_per_indent=2):
    """
    Fixes extra indentation in a given Python code string.

    :param code: String containing the Python code with extra indentation.
    :param spaces_per_indent: Number of spaces per indent level. Default is 4.
    :return: String with corrected indentation.
    """
    lines = code.split("\n")
    fixed_lines = []

    # find if most indents have 4 spaces or 8
    num_spaces = Counter(
        [len(line) - len(line.lstrip()) for line in lines]
    ).most_common()[0][0]
    if num_spaces == 4:
        return code
    elif num_spaces == 8:
        spaces_per_indent = 2

    for line in lines:
        stripped_line = line.lstrip()  # Remove leading whitespace
        # Calculate the number of leading spaces removed
        leading_spaces = len(line) - len(stripped_line)
        # Calculate the correct number of leading spaces
        corrected_leading_spaces = leading_spaces // spaces_per_indent
        # Reconstruct the line with fixed indentation
        fixed_line = " " * corrected_leading_spaces + stripped_line
        fixed_lines.append(fixed_line)

    # Join the fixed lines back into a single string
    fixed_code = "\n".join(fixed_lines)
    return fixed_code


def parse_llm_output(raw_llm_output: str) -> str:
    # Use regular expression to extract the function up to the 'return' statement
    # The pattern excludes content outside the function's scope and stops at 'return'
    # Splitting the input string into lines
    lines = raw_llm_output.split("\n")

    # Variables to track whether the function parsing has started, the parsed function,
    # and if the return statement was found
    parsing = False
    parsed_llm_out = ""
    return_found = False
    triple_quotes = False

    # Looping through each line to parse the function
    for line_idx, line in enumerate(lines):
        # Checking if the line is the start of the function definition
        if line.strip().startswith("def compute_reward("):
            parsing = True

        # avoiding Syntax error by discarding comments under """..."""
        if '"""' in line:
            if not triple_quotes:
                triple_quotes = np.max(
                    [
                        True if '"""' in next_line else False
                        for next_line in lines[line_idx:]
                    ]
                )
            else:
                triple_quotes = False
            continue
        if triple_quotes:
            continue

        # If we are currently parsing the function, append the line to the function string
        if parsing:
            # removing incorrect indent (gpt-4 sometimes adds incorrect indents leading to Indentation errors)
            if "def compute_reward" in line:
                line = line.strip()

            parsed_llm_out += line + "\n"
            # If the line contains a return statement, and this is not the final return
            # TODO: get the final return statement
            if "return" in line and (
                "```" in lines[line_idx : line_idx + 3]
                or "'''" in lines[line_idx : line_idx + 3]
            ):
                return_found = True

        # If we have found the return statement and reach a line that could indicate
        # the end of the function, stop parsing
        if return_found and (line.strip() == "" or not line.strip().startswith("    ")):
            parsed_llm_out = fix_indentation(parsed_llm_out)
            break

    return parsed_llm_out


def save_reward_string(
    rew_func_str: str,
    model_name: str,
    group_id: int,
    it: int,
    counter: int,
    baseline: str,
) -> str:
    print(
        f"\nSaving Reward String for Model: {model_name} | Iteration: {it} | Generation: {counter}.\n"
    )
    rewards_save_path = os.path.join(
        os.environ["ROOT_PATH"],
        f"{baseline}_database/{model_name}/group_{group_id}/reward_fns",
    )
    if not os.path.exists(rewards_save_path):
        os.makedirs(rewards_save_path)
    reward_filename = os.path.join(rewards_save_path, f"{it}_{counter}.txt")
    with open(reward_filename, "w") as infile:
        infile.write(rew_func_str)
    return reward_filename


def save_reward_string_new_envs(
    rew_func_str: str,
    model_name: str,
    group_id: int,
    it: int,
    counter: int,
    baseline: str,
    task_code_string: str,
    args: List[str],
) -> str:
    print(
        f"\nSaving Reward String for Model: {model_name} | Iteration: {it} | Generation: {counter}.\n"
    )
    rewards_save_path = os.path.join(
        os.environ["ROOT_PATH"],
        f"{baseline}_database/{model_name}/group_{group_id}/reward_fns",
    )
    if not os.path.exists(rewards_save_path):
        os.makedirs(rewards_save_path)
    reward_filename = os.path.join(rewards_save_path, f"{it}_{counter}.txt")

    gpt_reward_signature = "compute_reward" + "(self." + ", self.".join(args) + ")"
    reward_signature = [
        f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
        f"self.extras['gpt_reward'] = self.rew_buf.mean()",
        f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
    ]
    indent = " " * 8
    reward_signature = "\n".join([indent + line for line in reward_signature])
    if "def compute_reward(self)" in task_code_string:
        task_code_string_iter = task_code_string.replace(
            "def compute_reward(self):",
            "def compute_reward(self):\n" + reward_signature,
        )
    elif "def compute_reward(self, actions)" in task_code_string:
        task_code_string_iter = task_code_string.replace(
            "def compute_reward(self, actions):",
            "def compute_reward(self, actions):\n" + reward_signature,
        )
    else:
        raise NotImplementedError

    with open(reward_filename, "w") as infile:
        infile.writelines(task_code_string_iter + "\n")
        infile.writelines("from typing import Tuple, Dict" + "\n")
        infile.writelines("import math" + "\n")
        infile.writelines("import torch" + "\n")
        infile.writelines("from torch import Tensor" + "\n")
        if "@torch.jit.script" not in rew_func_str:
            code_string = "@torch.jit.script\n" + rew_func_str
        infile.writelines(code_string + "\n")
        # infile.write(rew_func_str)
    return reward_filename


def filter_traceback(s):
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
    return ""  # Return an empty string if no Traceback is found


def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = open(rl_filepath, "r").read()
        if "fps step:" in rl_log or "Traceback" in rl_log:
            # if log_status and "fps step:" in rl_log:
            #     logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            # if log_status and "Traceback" in rl_log:
            #     logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break


def save_fitness_score(
    fitness_score: float,
    model_name: str,
    group_id: int,
    it: int,
    counter: int,
    baseline: str,
) -> str:
    print(
        f"\nSaving Fitness Score for Model: {model_name} | Iteration: {it} | Generation: {counter}.\n"
    )
    if "auto" not in baseline:
        fitness_scores_str = "fitness_scores"
    elif "auto" in baseline:
        fitness_scores_str = "fitness_scores_auto"
    fitness_score_save_path = os.path.join(
        os.environ["ROOT_PATH"],
        f"{baseline}_database/{model_name}/group_{group_id}/{fitness_scores_str}",
    )
    if not os.path.exists(fitness_score_save_path):
        os.makedirs(fitness_score_save_path)
    score_filename = os.path.join(fitness_score_save_path, f"{it}_{counter}.txt")
    with open(score_filename, "w") as infile:
        infile.write(str(fitness_score))
    return score_filename


def save_human_feedback(
    human_feedback: str,
    model_name: str,
    group_id: int,
    it: int,
    counter: int,
    baseline: str,
) -> str:
    print(
        f"\nSaving Human Feedback for Model: {model_name} | Iteration: {it} | Generation: {counter}.\n"
    )
    feedback_save_path = os.path.join(
        os.environ["ROOT_PATH"],
        f"{baseline}_database/{model_name}/group_{group_id}/human_feedback",
    )
    if not os.path.exists(feedback_save_path):
        os.makedirs(feedback_save_path)
    feedback_filename = os.path.join(feedback_save_path, f"{it}_{counter}.txt")
    with open(feedback_filename, "w") as infile:
        infile.write(human_feedback)
    return feedback_filename


def format_human_feedback(human_feedback: str) -> str:
    pos_feedback, neg_feedback = human_feedback.split("\n")
    pos_feedback.replace("&&", "&")
    neg_feedback.replace("&&", "&")
    final_str = ""
    if pos_feedback != "":
        final_str += f"The satisfactory aspects are {pos_feedback}."
    if neg_feedback != "":
        final_str += f"Aspects that need improvement are {neg_feedback}."
    return final_str


def serialize_dict(dictionary, num_elements=10):
    ret_str = ""
    for key, values in dictionary.items():
        if isinstance(values, list) and len(values) > num_elements:
            step_size = (len(values) - 1) / (
                num_elements - 1
            )  # Adjust step_size to include the last element
            sampled_values = [
                values[round(i * step_size)] for i in range(num_elements - 1)
            ]
            sampled_values.append(values[-1])  # Explicitly add the last element
            # Remove potential duplicate if the second last element is the same due to rounding
            sampled_values = list(dict.fromkeys(sampled_values))
            ret_str += f"{key}: {sampled_values}\n"
        else:
            ret_str += f"{key}: {values}\n"
    return ret_str


class InvalidFunctionError(Exception):
    """Custom Error for reward"""

    def __init__(self, message):
        super().__init__(message)


def validate_callable_no_signature(func_str: str):
    # Look for "return" statements in the source code
    return_statements = [
        line.strip()
        for line in func_str.splitlines()
        if line.strip().startswith("return")
        and len(line.split(",")) == 2  # total reward, reward_components
    ]
    return return_statements


def linear_decay(
    iteration: int, initial_temp: float, final_temp: float, num_iterations: int
):
    """defines a temperature schedule for sampling of islands and individuals"""
    return initial_temp - (initial_temp - final_temp) * iteration / num_iterations


def cosine_annealing(
    iteration: int, initial_temp: float, final_temp: float, num_iterations: int
):
    """defines a temperature schedule for sampling of islands and individuals"""
    return final_temp + 0.5 * (initial_temp - final_temp) * (
        1 + np.cos(np.pi * iteration / num_iterations)
    )


def load_environment(env_choice: str, **kwargs):
    """
    Load the appropriate environment class dynamically.
    :param env_choice: The environment choice from the configuration ("HumanoidEnv" or "AdroitHandDoorEnv").
    :param kwargs: Additional arguments to pass to the environment constructor.
    :return: An instance of the selected environment.
    """
    env_map = {
        "HumanoidEnv": "rl_agent.HumanoidEnv.HumanoidEnv",
        "AdroitHandDoorEnv": "rl_agent.AdroitEnv.AdroitHandDoorEnv",
    }

    if env_choice not in env_map:
        raise ValueError(
            f"Unsupported environment choice: {env_choice}. Must be one of {list(env_map.keys())}."
        )

    # Import and load the class dynamically
    module_path, class_name = env_map[env_choice].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    env_class = getattr(module, class_name)

    # Instantiate the environment with the provided kwargs
    return env_class(**kwargs)
