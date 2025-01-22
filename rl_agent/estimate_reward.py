import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_max_reward_step(log_dir):
    event_files = glob.glob(
        os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True
    )
    max_reward = float("-inf")
    max_step = None

    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        # Check possible scalar tags that might store the reward
        possible_tags = ["eval/mean_reward", "train/mean_reward", "rollout/ep_rew_mean"]

        for tag in possible_tags:
            if tag in event_acc.Tags().get("scalars", []):
                for event in event_acc.Scalars(tag):
                    if event.value > max_reward:
                        max_reward = event.value
                        max_step = event.step

    return max_step, max_reward


def find_closest_model(model_dir, target_step):
    model_files = glob.glob(os.path.join(model_dir, "SAC_*.zip"))
    model_steps = [
        int(os.path.basename(f).split("_")[1].split(".")[0]) for f in model_files
    ]
    closest_step = min(model_steps, key=lambda x: abs(x - target_step))
    closest_model = f"SAC_{closest_step}.zip"
    return closest_model


log_dir = "/home/alkis/RLMujoco/adroit_door/logs_humanoid_standup/SAC_0"

model_dir = "/home/alkis/RLMujoco/adroit_door/models_humanoid_standup"
step, reward = get_max_reward_step(log_dir)
if step is not None:
    print(f"Highest reward of {reward} at step {step}")
    closest_model = find_closest_model(model_dir, step)
    print(f"Closest model to step {step} is {closest_model}")
else:
    print("No reward data found in the logs.")


def find_closest_model(model_dir, target_step):
    model_files = glob.glob(os.path.join(model_dir, "SAC_*.zip"))
    model_steps = [
        int(os.path.basename(f).split("_")[1].split(".")[0]) for f in model_files
    ]
    closest_step = min(model_steps, key=lambda x: abs(x - target_step))
    closest_model = f"SAC_{closest_step}.zip"
    return closest_model
