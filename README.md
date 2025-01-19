# REvolve: Reward Evolution with Large Language Models using Human Feedback
******************************************************

<p align="center">
    <a href="https://rishihazra.github.io/REvolve/" target="_blank">
        <img alt="Documentation" src="https://img.shields.io/website/https/rishihazra.github.io/EgoTV?down_color=red&down_message=offline&up_message=link">
    </a>
    <a href="https://arxiv.org/abs/2406.01309" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2406.01309-red">
    </a>
    <a href="https://arxiv.org/pdf/2406.01309">
        <img src="https://img.shields.io/badge/Downloads-PDF-blue">
    </a>
</p>

<p align="center">
  <img src="revolve.gif" alt="egoTV">
</p>

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
