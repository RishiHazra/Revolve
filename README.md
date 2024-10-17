# REvolve: Reward Evolution with Large Language Models using Human Feedback

## Setup
```shell
pip install -r requirements
```
For AirSim, follow the instruction on this link [https://microsoft.github.io/AirSim/build_linux/](AirSim)

## Run
```shell
python revolve.py  # for running REvolve

python eureka.py   # for Eureka baseline

python revolve_auto.py  # for REvolve Auto (with automatic feedback)

python eureka_auto.py  # Eureka Auto (Eureka with automatic feedback); set num_generate=1 in main() for Text2Rewards (T2R)  
```

## Other Utilities
* The prompts are listed in ```prompts``` folder.
* Elo scoring in ```human_feedback``` folder
