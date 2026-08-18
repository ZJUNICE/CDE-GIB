# Paper

This repository corresponds to the following paper. We are sharing the codes under the condition that reproducing full or part of codes must cite the paper.

> Ziqiong Wang, Xiaoxue Yu, Rongpeng Li, and Zhifeng Zhao, “Consensus-driven event-based graph information bottleneck for integrated communication and control,” *IEEE Trans. Veh. Techn.*, Apr. 2026
>

> Abstract: The integration of communication and control presents a critical challenge for Unmanned Aerial Vehicle (UAV) swarms operating in the formation control scenario. This co-design problem is typically formulated within the Multi-Agent Reinforcement Learning (MARL) framework. However, partial observability limitations will impair collaboration effectiveness, and a potential solution is to establish consensus through well-calibrated latent variables obtained from neighboring agents. Nevertheless, the rigid transmission of less informative content can still result in redundant information exchanges. Therefore, we propose a Consensus-Driven Event-Based Graph Information Bottleneck (CDE-GIB) method, which integrates the communication graph and information flow through a GIB regularizer to extract more concise message representations while avoiding the high computational complexity of inner-loop operations. Meanwhile, we investigate the potential impact of consensus errors induced by noise arising from multi-agent interactions. To further minimize the communication volume required for establishing consensus during interactions, we also develop a variable-threshold event-triggering mechanism. By simultaneously considering historical data and current observations, this mechanism capably evaluates the importance of information to determine whether an event should be triggered. Experimental results demonstrate that our proposed method outperforms existing state-of-the-art methods in terms of efficiency, adaptability, and noise robustness.

Note that this is a research project and, by definition, is unstable. Please write to us if you find something not correct or strange. 

## Project Structure

```text
CDE-GIB/
└── multiagent-particle-envs/
    ├── train.py                         # Main training entry point
    ├── eval.py                          # Evaluation entry point
    ├── mappo/
    │   ├── config.py                    # Training arguments and hyperparameters
    │   ├── sp_env.py                    # MPE wrapper used by MAPPO
    │   ├── algorithms/                  # MAPPO policy, actor, critic, and network modules
    │   ├── runner/                      # Training runners
    │   └── utils/                       # Replay buffers and utility functions
    └── multiagent/
        ├── core.py                      # MPE world, agent, and landmark definitions
        ├── environment.py               # MPE simulation logic
        └── scenarios                    # Formation-control scenario
```

## Installation


```
conda create -n formation 
conda activate formation
cd CDE-GIB/multiagent-particle-envs
pip install -e .
```


## Training

First enter the code directory:

```
cd CDE-GIB/multiagent-particle-envs
```

Run a training job:

```
python train.py
```
