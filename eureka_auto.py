import os
import random
import sys

import hydra

# os.environ["ROOT_PATH"] = os.getcwd()
sys.path.append(os.environ['ROOT_PATH'])
from rewards_database import EurekaDatabase
from modules import *
from utils import *
import prompts
from rl_agent.environment import CustomEnvironment


def is_valid_reward_fn(rew_fn, args):
    if rew_fn is None or args is None:
        raise InvalidRewardError
        # return False
    env_state = CustomEnvironment().env_state
    env_vars = env_state.keys()
    # check if all args are valid env args
    if set(args).intersection(set(env_vars)) != set(args):
        raise InvalidRewardError
    # return True


def generate_valid_reward(reward_generation: RewardFunctionGeneration,
                          in_context_prompt: str) -> [str, List[str]]:
    """
    single reward function generation until valid
    :param reward_generation: igeneratenitialized class of RewardFunctionGeneration from pipeline
    :param in_context_prompt: in context prompt used by the LLM to generate the new reward fn
    :return: return valid reward function string
    """
    while True:
        try:
            rew_func_str = reward_generation.generate_rf(in_context_prompt)
            rew_func, args = define_function_from_string(rew_func_str)
            is_valid_reward_fn(rew_func, args)
            print("\nValid reward function generated.\n")
            # saving reward string
            # reward_filename = save_reward_string(rew_func_str, model_name, it, counter)
            break  # Exit the loop if successful
        except (IndentationError, SyntaxError, TypeError, InvalidRewardError, AssertionError) as e:
            print(f"\nSpecific error caught: {e}\n")
        except Exception as e:  # Catch-all for any other exceptions
            print(f"\nUnexpected error occurred: {e}\n")
        print("Attempting to generate a new reward function due to an error.")
    return rew_func_str, args


def main():
    num_iterations = 5
    num_generate = 16
    operator = 'mutation_auto'
    model_name = 'gpt-4'
    group_id = 0  # just one group
    baseline = 'eureka_auto'
    num_gpus = 16

    system_prompt = prompts.types["system_prompt"]
    env_input_prompt = prompts.types["env_input_prompt"]

    reward_generation = RewardFunctionGeneration(system_prompt=system_prompt,
                                                 env_input=env_input_prompt,
                                                 model_name=model_name)
    reward_evaluation = RewardFunctionEvaluation(evaluator_type='fitness_function',
                                                 baseline=baseline)
    for it in range(0, num_iterations):
        # load all groups if it > 0
        eureka_database = EurekaDatabase(model_name=model_name,
                                         load_population=False if it == 0 else True,
                                         baseline=baseline)

        policies, valid_rew_func_strs, valid_rew_args = [], [], []
        # counter = 0  # reset counter after each iteration
        # max parallel training = num_queries x num_generate
        if it < 1:  # initially, uniformly populate the groups
            in_context_samples = (None, None)
            operator_prompt = ''
        else:
            operator_prompt = prompts.types[operator]
            in_context_samples = eureka_database.sample_in_context()
            # each sample in 'in_context_samples' is a tuple of (reward_fn_path: str, fitness_score: float)
        in_context_prompt = RewardFunctionGeneration.prepare_in_context_prompt(in_context_samples,
                                                                               operator_prompt,
                                                                               evolve=True if it >= 1 else False,
                                                                               baseline=baseline)
        # num_generate rw fns are generated for the same query (group_id is fixed)

        for counter in range(num_generate):
            print(f"\nGenerating reward function for counter: {counter}\n")  # Debugging print
            # generate valid reward fn str
            rew_func_str, args = generate_valid_reward(reward_generation, in_context_prompt)
            reward_filename = save_reward_string(rew_func_str, model_name, it,
                                                 group_id, counter, baseline)  # save reward fn str

            # initialize Driving Agent policy with the generated reward function
            policies.append(TrainPolicy(reward_filename, it, counter, num_gpus, group_id,
                                        port, model_name, baseline))
            valid_rew_func_strs.append(rew_func_str)
            valid_rew_args.append(args)

        # if no valid reward fn
        if len(policies) == 0:
            print("\n0 len policies\n")
            continue
        # train policies
        train_policies_in_parallel(policies)
        print("\nfinished training\n")

        # reward reflection
        valid_reward_fn_paths = []
        fitness_scores = []
        for i, policy in enumerate(policies):
            base_dir = os.path.join(os.environ['ROOT_PATH'],
                                    f"{baseline}_database/{model_name}/group_{group_id}")
            filename_suffix = f"{it}_{policy.counter}"
            reward_history_filename = os.path.join(base_dir, f"reward_history/{filename_suffix}.json")

            if os.path.exists(reward_history_filename):
                reward_dict = reward_evaluation.generate_behavior(reward_history_filename)
                # rewards_database.add_reward_to_group(valid_rew_funcs, valid_rew_args, avg_reward_scores, group_id)
                print(f"\nReward dict for group {group_id}, iteration {it}, "
                      f"counter {policy.counter}: {reward_dict}\n")
                fitness_score = reward_evaluation.evaluate_behavior(port, reward_dict, policy.counter, it, group_id)
                # fitness_score = generate_behaviour(reward_dict, counter, it, group_id)

                save_fitness_score(fitness_score, model_name, group_id, it, policy.counter, baseline)
                reward_fn_filename = os.path.join(base_dir, f"reward_fns/{filename_suffix}.txt")
                valid_reward_fn_paths.append(reward_fn_filename)
                fitness_scores.append(fitness_score)
            else:
                print(f"\nRewards history file not found for group {group_id}, "
                      f"iteration {it}, counter {policy.counter}\n")

        # store reward functions only if it improves initialized groups
        # for initialization, we don't use this step
        if it > 0:
            eureka_database.update_rewards(valid_reward_fn_paths, fitness_scores)


if __name__ == '__main__':
    main()
