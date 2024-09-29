# :tada: _T-ITS ACCEPTED!_ :confetti_ball:

# :page_with_curl: Safe Human-in-the-loop RL (SafeHiL-RL) with Shared Control for End-to-End Autonomous Driving

# :fire: Source Code Released! :fire:

## [[**T-ITS**]](https://ieeexplore.ieee.org/document/10596046) | [[**arXiv**]](https://www.researchgate.net/publication/382212078_Safety-Aware_Human-in-the-Loop_Reinforcement_Learning_With_Shared_Control_for_Autonomous_Driving)

:dizzy: As a **_pioneering work considering guidance safety_** within the human-in-the-loop RL paradigm, this work introduces a :fire: **_curriculum guidance mechanism_** :fire: inspired by the pedagogical principle of whole-to-part patterns in human education, aiming to standardize the intervention process of human participants.

:red_car: SafeHil-RL is designed to prevent **_policy oscillations or divergence_** caused by **_inappropriate or degraded human guidance_** during interventions using the **_human-AI shared autonomy_** technique, thereby improving learning efficiency, robustness, and driving safety.

:wrench: Realized in SMARTS simulator with Ubuntu 20.04 and Pytorch. 

Email: wenhui001@e.ntu.edu.sg

# Framework

<p align="center">
<img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/framework.png" height= "450" width="900">
</p>

# Frenet-based Dynamic Potential Field (FDPF)
<p float="left">
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_scenarios.png" height= "140" />
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_bound.png" height= "140" /> 
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_obstacle.png" height= "140" />
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_final.png" height= "140" />
</p>

# Demonstration (accelerated videos)

## Lane-change Performance
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/690b4b44-ac57-4ce1-890b-57ac125cef63
## Uncooperative Road User
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/52b2ec4b-8cd4-4b9d-a3a9-70bbd3b77157
## Cooperative Road User
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/02f95274-80cc-4e6b-8a5b-edfcbbd4d0a6
## Unobserved Road Structure
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/bb493f9c-d2c9-4db5-b034-ad456ef96c8a

# User Guide

## Clone the repository.
cd to your workspace and clone the repo.
```
git clone https://github.com/OscarHuangWind/Safe-Human-in-the-Loop-RL.git
```

## Create a new Conda environment.
cd to your workspace:
```
conda env create -f environment.yml
```

## Activate virtual environment.
```
conda activate safehil-rl
```

## Install Pytorch
Select the correct version based on your cuda version and device (cpu/gpu):
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Install the SMARTS.
```
# Download SMARTS

git clone https://github.com/huawei-noah/SMARTS.git

cd <path/to/SMARTS>

# Important! Checkout to comp-1 branch
git checkout comp-1

# Install the system requirements.
bash utils/setup/install_deps.sh

# Install smarts.
pip install -e '.[camera_obs,test,train]'

# Install extra dependencies.
pip install -e .[extras]
```

## Build the scenario.
```
cd <path/to/Safe-Human-in-the-loop-RL>
scl scenario build --clean scenario/straight/
```

## Visulazation
```
scl envision start
```
Then go to http://localhost:8081/

## Training
Modify the sys path in **main.py** file, and run:
```
python main.py
```

## Human Guidance
Change the model in **main.py** file to SaHiL/PHIL/HIRL, and run:
```
python main.py
```
Check the code in keyboard.py to get idea of keyboard control.

Alternatively, you can use G29 set to intervene the vehicle control, check the lines from 177 to 191 in main.py file for the details.

The "Egocentric View" is recommended for the human guidance.

## Evaluation
Edit the mode in config.yaml as evaluation and run:
```
python main.py
```




