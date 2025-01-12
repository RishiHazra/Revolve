import os
import sys

sys.path.append(os.environ["ROOT_PATH"])
from rewards_database import RevolveDatabase, EurekaDatabase
from modules import *
import utils
import random
import prompts
from evolutionary_utils.custom_environment import CustomEnvironment
from typing import Callable, List
import absl.logging as logging
from functools import partial

import hydra


def is_valid_reward_fn(generated_fn: Callable, generated_fn_str: str, args: List[str]):
    """validate generated heuristic function"""
    if generated_fn is None or args is None:
        raise utils.InvalidFunctionError("Generated function has no arguments.")
    env_state = CustomEnvironment().env_state
    env_vars = env_state.keys()
    # check if all args are valid env args
    if set(args).intersection(set(env_vars)) != set(args):
        raise utils.InvalidFunctionError("Generated function uses invalid arguments.")
    # TODO: test the following for REvolve
    # Get the return type annotation
    return_statements = utils.validate_callable_no_signature(generated_fn_str)
    if not return_statements:
        raise utils.InvalidFunctionError(
            "The function does not have any return statements."
        )
    return True


def generate_valid_reward(
    reward_generation: RewardFunctionGeneration,
    in_context_prompt: str,
    max_trials: int = 10,
) -> [str, List[str]]:
    """
    single function generation until valid
    :param reward_generation: initialized class of RewardFunctionGeneration
    :param in_context_prompt: in context prompt used by the LLM to generate the new fn
    :param max_trials: maximum number of trials to generate
    :return: return valid function string
    """
    # used in case we want to provide python error feedbacks to the LLM
    error_feedback = ""
    error_flag = False
    trials = 0
    while True:
        try:
            rew_func_str = reward_generation.generate_rf(
                in_context_prompt + error_feedback
            )
            rew_func, args = utils.define_function_from_string(rew_func_str)
            is_valid_reward_fn(rew_func, rew_func_str, args)
            logging.info("Valid reward function generated.")
            error_flag = False
            error_feedback = ""
            break  # Exit the loop if successful
        except Exception as e:
            logging.info(f"Specific error caught: {e}")
            error = e
            error_flag = True
        logging.info("Attempting to generate a new function due to an error.")
        trials += 1
        if trials >= max_trials:
            logging.info("Exceeded max trials.")
            return None, None
    return rew_func_str, args


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.environ["ROOT_PATH"], "cfg"),
    config_name="generate",
)
def main(cfg):
    system_prompt = prompts.types["system_prompt"]
    env_input_prompt = prompts.types["env_input_prompt"]

    reward_generation = RewardFunctionGeneration(
        system_prompt=system_prompt, env_input=env_input_prompt
    )

    # create log directory
    log_dir = cfg.data_paths.output_logs
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tracker = utils.DataLogger(os.path.join(log_dir, "progress.log"))

    # define a schedule for temperature of sampling
    temp_scheduler = partial(
        utils.linear_decay,
        initial_temp=cfg.database.initial_temp,
        final_temp=cfg.database.final_temp,
        num_iterations=cfg.evolution.num_generations,
    )
    if "revolve" in cfg.evolution.baseline:
        database = partial(
            RevolveDatabase,
            num_islands=cfg.database.num_islands,
            max_size=cfg.database.max_island_size,
            crossover_prob=cfg.database.crossover_prob,
            migration_prob=cfg.database.migration_prob,
            reward_fn_dir=cfg.database.rewards_dir,
            baseline=cfg.evolution.baseline,
        )
    else:
        database = partial(
            EurekaDatabase,
            num_islands=1,  # Eureka for a single island
            max_size=cfg.database.max_island_size,
            reward_fn_dir=cfg.database.rewards_dir,
            baseline=cfg.evolution.baseline,
        )

    for iteration_id in range(1, cfg.evolution.num_generations + 1):
        # fix the temperature for sampling
        temperature = temp_scheduler(iteration=iteration_id)
        print(
            f"\n========= Generation {iteration_id} | Model: {cfg.evolution.baseline} | temperature: {round(temperature, 2)} =========="
        )
        # load all groups if iteration_id > 0, else initialize empty islands
        rewards_database = database(load_islands=not iteration_id == 0)

        rew_fn_strings = []  # valid rew fns
        # fitness_scores = []
        island_ids = []
        counter_ids = []
        # metrics_dicts = []
        policies = []

        # for each generation, produce new individuals via mutation or crossover
        for counter_id in range(cfg.evolution.individuals_per_generation):
            if iteration_id == 0:  # initially, uniformly populate the islands
                # TODO: to avoid corner cases, populate all islands uniformly
                island_id = random.choice(range(rewards_database.num_islands))
                in_context_samples = (None, None)
                operator_prompt = ""
            else:  # gen_id > 0: start the evolutionary process
                (
                    in_context_samples,
                    island_id,
                    operator,
                ) = rewards_database.sample_in_context(
                    cfg.few_shot, temperature
                )  # weighted sampling of islands and corresponding individuals
                operator = f'{operator}_auto' if 'auto' in cfg.evolution.baseline else operator
                operator_prompt = prompts.types[operator]

            island_ids.append(island_id)
            # each sample in 'in_context_samples' is a tuple of (fn_path: str, fitness_score: float)
            in_context_prompt = RewardFunctionGeneration.prepare_in_context_prompt(
                in_context_samples,
                operator_prompt,
                evolve=iteration_id > 0,
                baseline=cfg.evolution.baseline,
            )
            logging.info(f"Designing reward function for counter {counter_id}")
            # generate valid fn str
            rew_func_str, _ = generate_valid_reward(
                reward_generation, in_context_prompt
            )
            try:
                # initialize RL agent policy with the generated reward function
                policies.append(
                    TrainPolicy(
                        rew_func_str,
                        iteration_id,
                        counter_id,
                        island_id,
                        cfg.evolution.baseline,
                        cfg.database.rewards_dir,
                    )
                )
                rew_fn_strings.append(rew_func_str)
                counter_ids.append(counter_id)
            except:
                logging.info(
                    "Oops, something broke again :( Let's toss it out the window and call it modern art!"
                )
                continue

        # run policies in parallel
        # if no valid reward fn
        if len(policies) == 0:
            logging.info("No valid reward functions. Hence, no policy trains required.")
            continue
        # train policies in parallel
        logging.info(f"Training {len(policies)} policies in parallel.")
        ckpt_paths = train_policies_in_parallel(policies)
        logging.info("Policy training finished.")

        # evaluate performance for generated reward functions
        logging.info("Evaluating trained policies in parallel.")
        metrics_dicts = evaluate_policies_in_parallel(ckpt_paths)
        fitness_scores = [metric_dict["fitness"] for metric_dict in metrics_dicts]
        logging.info("Evaluation finished.")

        # store individuals only if it improves overall island fitness
        # for initialization, we don't use this step
        if iteration_id > 0:
            rewards_database.add_individuals_to_islands(
                [iteration_id] * len(island_ids),
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                metrics_dicts,
                island_ids,
            )
        else:  # initialization step (generation = 0)
            rewards_database.seed_islands(
                [iteration_id] * len(island_ids),
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                metrics_dicts,
                island_ids,
            )

        island_info = [
            {
                island_id: {
                    f"{gen_id}_{count_id}": fitness
                    for gen_id, count_id, fitness in zip(
                        island.generation_ids, island.counter_ids, island.fitness_scores
                    )
                }
            }
            for island_id, island in enumerate(rewards_database._islands)
        ]
        tracker.log({"generation": iteration_id, "islands": island_info})


if __name__ == "__main__":
    main()
