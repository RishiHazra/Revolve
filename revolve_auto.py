import os
import random
import sys

import hydra

sys.path.append(os.environ['ROOT_PATH'])
from rewards_database import RewardsDatabase
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
    :param reward_generation: initialized class of RewardFunctionGeneration from pipeline
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


@hydra.main(config_path='cfg',
            config_name='generate')
def main(cfg):
    generate = cfg["generate"]
    num_gpus = generate.num_gpus
    num_iterations = generate.num_iterations

    # rewards database
    rewardDatabase_cfg = cfg['rewardDatabase']
    num_groups = rewardDatabase_cfg.num_groups
    max_group_size = rewardDatabase_cfg.max_group_size
    # reset_period = rewardDatabase_cfg.reset_period
    crossover_prob = rewardDatabase_cfg.crossover_prob

    baseline = 'revolve_auto'

    # llm reward generation
    model_name = generate.model_name
    system_prompt = prompts.types["system_prompt"]
    env_input_prompt = prompts.types["env_input_prompt"]

    reward_generation = RewardFunctionGeneration(system_prompt=system_prompt,
                                                 env_input=env_input_prompt,
                                                 model_name=model_name)
    reward_evaluation = RewardFunctionEvaluation(evaluator_type='fitness_function',
                                                 baseline=baseline)
    few_shot = cfg["few_shot"]
    for it in range(1, num_iterations):
        # load all groups if it > 0
        rewards_database = RewardsDatabase(num_groups=num_groups,
                                           max_size=max_group_size,
                                           crossover_prob=crossover_prob,
                                           model_name=model_name,
                                           load_groups=False if it == 0 else True,
                                           baseline=baseline)

        num_queries = generate.num_queries
        policies, valid_rew_func_strs, valid_rew_args, group_ids = [], [], [], []
        counter = 0  # reset counter after each iteration
        # max parallel training = num_queries x num_generate
        for _ in range(num_queries):
            if it < 1:  # initially, uniformly populate the groups
                group_id = random.choice(range(rewards_database.num_groups))
                in_context_samples = (None, None)
                operator_prompt = ''
            else:  # after k iterations, start the evolutionary process
                in_context_samples, group_id, operator = rewards_database.sample_in_context(few_shot)
                operator = f'{operator}_auto'
                operator_prompt = prompts.types[operator]
            # each sample in 'in_context_samples' is a tuple of (reward_fn_path: str, fitness_score: float)
            in_context_prompt = RewardFunctionGeneration.prepare_in_context_prompt(in_context_samples,
                                                                                   operator_prompt,
                                                                                   evolve=True if it >= 1 else False,
                                                                                   baseline=baseline)
            num_generate = generate.num_generate
            # num_generate rw fns are generated for the same query (group_id is fixed)
            for _ in range(num_generate):
                counter += 1
                print(f"\nGenerating reward function for counter: {counter}\n")  # Debugging print
                # generate valid reward fn str
                rew_func_str, args = generate_valid_reward(reward_generation, in_context_prompt)
                reward_filename = save_reward_string(rew_func_str, model_name, group_id, it,
                                                     counter, baseline)  # save reward fn str

                # initialize Driving Agent policy with the generated reward function
                policies.append(TrainPolicy(reward_filename, it, counter, num_gpus,
                                            group_id, port, model_name, baseline))
                # valid_rew_func_strs.append(rew_func_str)
                # valid_rew_args.append(args)
                group_ids.append(group_id)

        # if no valid reward fn
        if len(policies) == 0:
            print("\n0 len policies\n")
            continue
        # train policies
        train_policies_in_parallel(policies)
        print("\nfinished training\n")

        # reward reflection
        valid_reward_fn_paths = []
        valid_group_ids = []
        fitness_scores = []
        for i, (policy, group_id) in enumerate(zip(policies, group_ids)):
            base_dir = os.path.join(os.environ['ROOT_PATH'],
                                    f"{baseline}_database/{model_name}/group_{group_id}")
            filename_suffix = f"{it}_{policy.counter}"
            reward_history_filename = os.path.join(base_dir, f"reward_history/{filename_suffix}.json")

            if os.path.exists(reward_history_filename):
                reward_dict = reward_evaluation.generate_behavior(reward_history_filename)
                # rewards_database.add_reward_to_group(valid_rew_funcs, valid_rew_args, avg_reward_scores, group_id)
                print(f"\nReward dict for group {group_id}, iteration {it}, "
                      f"counter {policy.counter}: {reward_dict}\n")
                fitness_score = reward_evaluation.evaluate_behavior(reward_dict, policy.counter, it, group_id)
                # fitness_score = generate_behaviour(reward_dict, counter, it, group_id)

                save_fitness_score(fitness_score, model_name, group_id, it, policy.counter, baseline)
                reward_fn_filename = os.path.join(base_dir, f"reward_fns/{filename_suffix}.txt")
                valid_reward_fn_paths.append(reward_fn_filename)
                valid_group_ids.append(group_id)
                fitness_scores.append(fitness_score)
            else:
                print(f"\nRewards history file not found for group {group_id}, "
                      f"iteration {it}, counter {policy.counter}\n")

        # store reward functions only if it improves initialized groups
        # for initialization, we don't use this step
        if it > 0:
            rewards_database.add_reward_to_group(valid_reward_fn_paths,
                                                 fitness_scores,
                                                 valid_group_ids)


if __name__ == '__main__':
    main()