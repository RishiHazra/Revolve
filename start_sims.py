import subprocess
import os
import airsim
import time


def connect_car_client(sim_id, ip_address):
    client = airsim.CarClient(ip=ip_address)
    client.confirmConnection()
    print(f"Car client {sim_id} successfully connected to {ip_address}")
    time.sleep(2)



simulation_processes = []
num_sims = 25
num_gpus = 4
# Define the path to your .sh file
sh_file_path = os.environ["AIRSIMNH_PATH"]
# Define the path to your settings files
settings_path = os.environ["AIRSIM_PATH"]
settings_base_path = os.environ["AIRSIM_PATH"]


for sim_id in range(1, num_sims + 1):
   # gpu_id = sim_id % num_gpus

    #settings_file = f"{settings_path}/json_settings{sim_id}.json"
    settings_file_path = f"{settings_base_path}/settings_{sim_id}.json"

    print("settigns path",settings_path)
    #command = f"CUDA_VISIBLE_DEVICES={gpu_id} {sh_file_path} -windowed -RenderOffScreen -settings={settings_file}"
    command = f"{sh_file_path}  -RenderOffScreen -settings=\"{settings_file_path}\""
    print(f"Executing command: {command}")
    proc = subprocess.Popen(command, shell=True)
    simulation_processes.append(proc.pid)
    
