# REvolve: Reward Evolution with Large Language Models using Human Feedback
******************************************************
**Official code release of our ICLR 2025 paper.**

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
# clone the repository 
git clone https://github.com/RishiHazra/Revolve.git
cd Revolve
pip install -e .
```

## Run
```shell
export ROOT_PATH='Revolve'
export OPENAI_API_KEY='<your openai key>'
python main.py
```

*Note, we will soon release the AirSim environment setup script.*

For AirSim, follow the instruction on this link [https://microsoft.github.io/AirSim/build_linux/](AirSim)
```shell
$ export AIRSIM_PATH='AirSim'
$ export AIRSIMNH_PATH='AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh'
```

## Other Utilities
* The prompts are listed in ```prompts``` folder.
* Elo scoring in ```human_feedback``` folder

## Citation

### To cite our paper:
```bibtex
@misc{hazra2024revolverewardevolutionlarge,
      title={REvolve: Reward Evolution with Large Language Models using Human Feedback}, 
      author={Rishi Hazra and Alkis Sygkounas and Andreas Persson and Amy Loutfi and Pedro Zuidberg Dos Martires},
      year={2024},
      eprint={2406.01309},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2406.01309}, 
}
```
