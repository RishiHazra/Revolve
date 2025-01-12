# REvolve: Reward Evolution with Large Language Models using Human Feedback

## Setup
```shell
pip install -r requirements
```
For AirSim, follow the instruction on this link [https://microsoft.github.io/AirSim/build_linux/](AirSim)

```shell
$ export ROOT_PATH='Revolve'
$ export AIRSIM_PATH='AirSim'
$ export AIRSIMNH_PATH='AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh'
$ export OPENAI_API_KEY=''

```

## Run
```shell
python main.py  # for running REvolve
```

## Other Utilities
* The prompts are listed in ```prompts``` folder.
* Elo scoring in ```human_feedback``` folder
