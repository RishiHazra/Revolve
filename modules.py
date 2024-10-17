import concurrent.futures
import json
import os
import time
from typing import List, Tuple, Optional, Dict

# sys.path.append(os.environ['ROOT_PATH'])
import openai
from openai import OpenAI

from rl_agent.generate_scores import generate_behaviour
from rl_agent.main import run_training
from utils import parse_llm_output, serialize_dict, format_human_feedback

openai_api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


# generates reward functions
class RewardFunctionGeneration:
    def __init__(self, system_prompt: str, env_input: str, model_name: str):
        # TODO: change system message based on Eureka
        self.system_prompt = system_prompt
        self.env_input = env_input  # env_class + task
        # self.task = task
        if model_name == 'gpt-4':
            self.llm = 'gpt-4-1106-preview'
        elif model_name == 'gpt-3.5':
            self.llm = 'gpt-3.5-turbo-1106'

    def query_llm(self, in_context_prompt: str) -> Tuple[str, int, int]:
        response = client.chat.completions.create(model=self.llm,  # gpt-4-1106-preview, gpt-3.5-turbo-1106
                                                  messages=[
                                                      {"role": "system",
                                                       "content": self.system_prompt + "\n" + self.env_input},
                                                      {"role": "user", "content": f"{in_context_prompt}"},
                                                  ],
                                                  temperature=1,
                                                  max_tokens=4096,
                                                  top_p=1,
                                                  frequency_penalty=0,
                                                  presence_penalty=0)

        return (response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens)

    @staticmethod
    def prepare_in_context_prompt(in_context_samples: Optional[List[Tuple[str, float]]],
                                  operator_prompt: str, evolve: bool,
                                  baseline: str) -> str:
        # prepares a prompt from in context examples sampled from RewardsDatabase
        in_context_samples_str = ''
        if not evolve:
            return in_context_samples_str
        for filename, fitness_score in in_context_samples:
            in_context_samples_str += '\n\n```python\n'
            in_context_samples_str += open(filename, 'r').read()
            in_context_samples_str += '\n```\n'
            base_dir = '/'.join(filename.split('/')[:-2])  # root_path/database/gpt-4/group_*/
            iteration_id, model_id = filename.split('/')[-1].replace('.txt', '').split('_')
            reward_history_file = f"{base_dir}/reward_history/{iteration_id}_{model_id}.json"
            # fitness_score_file = f"{base_dir}/fitness_scores/{iteration_id}_{model_id}.txt"
            reward_history = json.load(open(reward_history_file, 'r'))
            # fitness_score = open(fitness_score_file, 'r').read()
            in_context_samples_str += f'fitness score: {fitness_score}'
            in_context_samples_str += f'\n{serialize_dict(reward_history)}'
            if 'auto' not in baseline:
                human_feedback_filename = f"{base_dir}/human_feedback/{iteration_id}_{model_id}.txt"
                human_feedback = open(human_feedback_filename, 'r').read()
                human_feedback = format_human_feedback(human_feedback)
                in_context_samples_str += f'\nhuman feedback: {human_feedback}'
        operator_prompt = operator_prompt.replace("\n\n<EXAMPLES>", in_context_samples_str)
        operator_prompt = operator_prompt.replace("<EPOCHS>", '10000')
        return operator_prompt

    def single_query(self, in_context_prompt: str) -> str:
        raw_llm_output, num_prompt_tokens, num_completion_tokens = self.query_llm(in_context_prompt)
        parsed_llm_output = parse_llm_output(raw_llm_output)
        return parsed_llm_output

    def generate_rf(self, in_context_prompt: str) -> str:
        parsed_rf = None
        while True:
            try:
                parsed_rf = self.single_query(in_context_prompt)
                break
            except openai.RateLimitError or openai.APIError or openai.Timeout:
                time.sleep(10)
                continue
        return parsed_rf


class TrainPolicy:
    def __init__(self, reward_fn_path: str, iteration: int, counter: int, num_gpus: int,
                 group_id: int, port: int,  llm_model: str, baseline: str):
        self.base_ip = "127.0.0."
        self.reward_fn_path = reward_fn_path
        # self.args = reward_fn_args
        #    self._load_train_cfg()
        # self.train_cfg = cfg
        #  self.driving_agent = DrivingAgent(i)
        self.num_gpus = num_gpus
        self.counter = counter  # generation index for iteration (multiple rewards generated per iteration
        # per query)
        self.call_count = 1
        self.gpu_id = None
        self.iteration = iteration
        self.group_id = group_id
        self.port = port
        self.llm_model = llm_model  # llm model used to generate reward fns
        self.baseline = baseline  # ['revolve', 'revolve_auto', 'eureka', 'eureka_auto']

    # def _load_train_cfg(self):
    #  config_path = os.path.join(os.environ['ROOT_PATH'], 'cfg')
    # hydra.initialize(config_path=config_path)
    # self.train_cfg = hydra.compose(config_name='train')

    def _generate_ip(self):
        # Generate a new IP address by incrementing the last part of the base IP
        new_ip = self.base_ip + str(self.counter)
        # Increment call count for the next call
        return new_ip

    def train_policy(self, model_idx: int):
        ip_address = self._generate_ip()

        run_training(ip_address, self.reward_fn_path, self.counter, self.iteration, self.group_id, self.port,
                     self.llm_model, self.baseline)

        # # Set the environment variable for CUDA device
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     try:
        #         tf.config.experimental.set_memory_growth(gpus[0], True)
        #     except RuntimeError as e:
        #         print(e)
        # ip_address = self._generate_ip()
        # print("Training on GPU:", gpu_id, "| IP Address:", ip_address)
        # run_training(ip_address, self.reward_fn_path, self.counter,
        #              self.iteration, self.group_id, self.port, self.llm_model, self.baseline)


# human evaluation, fitness functions
class RewardFunctionEvaluation:
    def _init_(self, evaluator_type: str, baseline: str):
        self.evaluator_type = evaluator_type  # human_feedback, fitness_function
        self.baseline = baseline
        self.base_ip = "127.0.0."

    def generate_behavior(self, filename: str) -> Dict:
        # be provided
        reward_history_dict = json.load(open(filename, 'r'))
        return reward_history_dict

    def _generate_ip(self):
        # Generate a new IP address by incrementing the last part of the base IP
        new_ip = self.base_ip + str(self.counter)
        # Increment call count for the next call
        return new_ip

    def evaluate_behavior(self, port, reward_history_dict: Dict, counter: int, it: int, group_id: int) -> float:
        # fitness_score = np.mean(np.array(reward_history_dict['total_reward'][-20:]))
        ip_adress = self._generate_ip()
        fitness_score = generate_behaviour(ip_adress, port, reward_history_dict, counter, it, group_id, self.baseline)
        return fitness_score


def train_policies_in_parallel(model_classes: List[TrainPolicy]):
    print("len model classes", len(model_classes))

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(model_classes)) as executor:
        print("inside pool")
        futures = [executor.submit(model_classes[i].train_policy, i) for i in range(len(model_classes))]
        #  results = executor.map(train_model, enumerate(model_classes))

        for future in futures:
            print("result", future.result())  # Check if there are any results or exceptions

    print("Training completed for all models")
