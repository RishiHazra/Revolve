import os
import sys

# os.environ["ROOT_PATH"] = os.getcwd()
sys.path.append(os.environ['ROOT_PATH'])
ISAAC_ROOT_DIR = os.path.join(os.environ['ROOT_PATH'], 'isaacgymenvs/isaacgymenvs')
from rewards_database import EurekaDatabase
from modules import *
from utils import *
import prompts
# from rl_agent.environment import CustomEnvironment
import subprocess


# def is_valid_reward_fn(rew_fn, args):
#     if rew_fn is None or args is None:
#         raise InvalidRewardError
#         # return False
#     env_state = CustomEnvironment().env_state
#     env_vars = env_state.keys()
#     # check if all args are valid env args
#     if set(args).intersection(set(env_vars)) != set(args):
#         raise InvalidRewardError
#     # return True


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
            # is_valid_reward_fn(rew_func, args)
            # TODO: do we need to add this extra check for new envs?
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
    env_parent = 'isaac'  # ['bidex', 'isaac']
    env_name = 'ant'  # ['humanoid', 'ant']
    task = 'Ant'  # ['Ant', 'Humanoid']
    max_episodes = 3000  # RL Policy training iterations (decrease this to make the feedback loop faster)

    system_prompt = prompts.types["system_prompt"]
    # env_input_prompt = prompts.types["env_input_prompt"]
    env_input_prompt = open(f'{os.environ["ROOT_PATH"]}/envs/{env_parent}/{env_name}_obs.py', 'r').read()
    task_code_string = open(f'{os.environ["ROOT_PATH"]}/envs/{env_parent}/{env_name}.py', 'r').read()

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
            reward_filename = save_reward_string_new_envs(rew_func_str, model_name, it,
                                                          group_id, counter, baseline,
                                                          task_code_string, args)  # save reward fn str

            # initialize Driving Agent policy with the generated reward function
            # Construct the command list
            # TODO: needs hydra to run: calls cfg from ISAAC_ROOT_DIR
            #  (use same hydra version as Eureka)
            base_dir = os.path.join(os.environ['ROOT_PATH'],
                                    f"{baseline}_database/{model_name}/group_{group_id}")
            filename_suffix = f"{it}_{counter}"
            reward_history_filename = os.path.join(base_dir, f"reward_history/{filename_suffix}.json")

            # rl_filepath = f"env_iter{iter}_response{counter}.txt"
            with open(reward_history_filename, 'w') as f:
                process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                                            'hydra/output=subprocess',
                                            f'task={task}',
                                            f'headless=True', f'capture_video=False',
                                            'force_render=False',
                                            f'max_iterations={max_episodes}'],
                                           stdout=f, stderr=f)
            block_until_training(reward_history_filename, log_status=True, iter_num=iter, response_id=counter)

            # Run the command using subprocess
            # policies.append(TrainPolicy(reward_filename, it, counter, num_gpus, group_id,
            #                             port, model_name, baseline))
            policies.append(process)
            valid_rew_func_strs.append(rew_func_str)
            valid_rew_args.append(args)

        # if no valid reward fn
        if len(policies) == 0:
            print("\n0 len policies\n")
            continue

        # train policies
        # TODO: add save_rewards_as_json(rewards_history_filename, rewards_history)
        for counter, policy in enumerate(policies):
            policy.communicate()
        #     rl_filepath = f"env_iter{iter}_response{counter}.txt"
        #     try:
        #         with open(rl_filepath, 'r') as f:
        #             stdout_str = f.read()
        #     except:
        #         continue
        #
        #     # content = ''
        #     traceback_msg = filter_traceback(stdout_str)
        #     if traceback_msg == '':
        #         # If RL execution has no error, provide policy statistics feedback
        #         exec_success = True
        #         lines = stdout_str.split('\n')
        #         for i, line in enumerate(lines):
        #             if line.startswith('Tensorboard Directory:'):
        #                 break
        #         tensorboard_logdir = line.split(':')[-1].strip()
        #         tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        #         print(tensorboard_logs)
                # max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                # epoch_freq = max(int(max_iterations // 10), 1)
                #
                # content += policy_feedback.format(epoch_freq=epoch_freq)

        # train_policies_in_parallel(policies)
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

        # store reward functions only if improves initialized groups
        # for initialization, we don't use this step
        if it > 0:
            eureka_database.update_rewards(valid_reward_fn_paths, fitness_scores)


if __name__ == '__main__':
    main()
