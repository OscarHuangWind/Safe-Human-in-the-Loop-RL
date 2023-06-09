# Human-in-the-loop-RL
This repository contains the illustrative diagrams and demonstration videos of the proposed approach called **safety-aware human-in-the-loop reinforcement learning (SaHiL-RL)**. 


:hourglass_flowing_sand: We will publish the source code once the paper is accepted. 

:beer: Prior to this, we are more than happy to discuss the details of our algorithm if you are interested. Please feel free to contact us without any hesitation.

Email: wenhui001@e.ntu.edu.sg

# Framework

<p align="center">
<img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/framework.png" height= "450" width="720">
</p>

# Frenet-based Dynamic Potential Field (FDPF)
<p float="left">
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_scenarios.png" height= "140" />
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_bound.png" height= "140" /> 
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_obstacle.png" height= "140" />
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_final.png" height= "140" />
</p>

# Demonstration

## Lane-change Performance
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/690b4b44-ac57-4ce1-890b-57ac125cef63
## Uncooperative Road User
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/52b2ec4b-8cd4-4b9d-a3a9-70bbd3b77157
## Cooperative Road User
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/02f95274-80cc-4e6b-8a5b-edfcbbd4d0a6
## Unobserved Road Structure
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/bb493f9c-d2c9-4db5-b034-ad456ef96c8a

# How to use

## Create a new Conda environment.
Specify your own name for the virtual environment, e.g., hil-rl:
```
conda create -n hil-rl python=3.7
```

## Activate virtual environment.
```
conda activate hil-rl
```

## Install Dependencies.
```
conda install gym==0.19.0
```

```
pip install cpprb tqdm pyyaml scipy matplotlib pandas casadi
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

# Install the system requirements.
bash utils/setup/install_deps.sh

# Install smarts.
pip install -e '.[camera_obs,test,train]'

# Install extra dependencies.
pip install -e .[extras]
```

## Build the scenario.
```
cd <path/to/Human-in-the-loop-RL>
scl scenario build --clean scenario/straight_with_left_turn/
```

## Visulazation
```
scl envision start
```
Then go to http://localhost:8081/

## Training
```
python main.py
```

## Evaluation
Edit the mode in config.yaml as evaluation and run:
```
python main.py
```




