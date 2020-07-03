# Branching Dueling Q-Network (BDQ)

<p align="center">
  <img src="../../data/bdq_network.png" alt="BDQ Network" width=80%">
</p>

Branching Dueling Q-Network (BDQ) is a novel agent which is based on the incorporation of the proposed [action branching architecture](https://arxiv.org/abs/1711.08946) into the [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236) algorithm, as well as adapting a selection of its extensions, [Double Q-Learning](https://arxiv.org/abs/1509.06461), [Dueling Network Architectures](https://arxiv.org/abs/1511.06581), and [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952). 

As we show in the [paper](https://arxiv.org/abs/1711.08946), BDQ is able to solve numerous continuous control domains via discretization of the action space. Most remarkably, we have shown that BDQ is able to perform well on the Humanoid-v1 domain with a total of 6.5 x 10<sup>25</sup> discrete actions.     

Our TensorFlow code for BDQ is based on the implementation of the DQN agent as part of the initial release of the OpenAI Baselines. However, it does not require installation of the Baselines as we have made it into a free-standing codebase with relative path imports.

<p align="center">
<img src="../../data/BDQ_Reacher3DOF-v0.gif" alt="Reacher3DOF-v0" width=24%/> 
<img src="../../data/BDQ_Reacher4DOF-v0.gif" alt="Reacher4DOF-v0" width=24%/> 
<img src="../../data/BDQ_Reacher5DOF-v0.gif" alt="Reacher5DOF-v0" width=24%/> 
<img src="../../data/BDQ_Reacher6DOF-v0.gif" alt="Reacher6DOF-v0" width=24%/>
<br>
<img src="../../data/BDQ_Reacher-v1.gif" alt="Reacher-v1" width=24%/> 
<img src="../../data/BDQ_Hopper-v1.gif" alt="Hopper-v1" width=24%/> 
<img src="../../data/BDQ_Walker2d-v1.gif" alt="Walker2d-v1" width=24%/> 
<img src="../../data/BDQ_Humanoid-v1.gif" alt="Humanoid-v1" width=24%/>
</p>


## Getting Started

You can clone this repository by:

```bash
git clone https://github.com/atavakol/action-branching-agents.git
``` 

To train a new model or evaluate a pre-trained one, change directory to the agent's main directory to run the corresponding scripts (required due to using relative paths).


### Train 

You can readily train a new model for any continuous control domain from the OpenAI Gym collection or the custom reaching domains provided in this repository, by running the [train_continuous.py](train_continuous.py) script from the agent's main directory. 

```bash
cd action-branching-agents/agents/bdq
python train_continuous.py 
```


### Evaluate 

Alternatively, you can evaluate a pre-trained model saved in the agent's `trained_models` directory, by running the [enjoy_continuous.py](enjoy_continuous.py) script from the agent's main directory. By default, evaluation is run using a greedy policy.


#### Provided pre-trained models

Currently, a set of pre-trained models are provided in the agent's `trained_models` directory for the following domains: 

* MuJoCo (custom): `Reacher3DOF-v0`, `Reacher4DOF-v0`, `Reacher5DOF-v0`, `Reacher6DOF-v0`
* MuJoCo (standard): `Reacher-v1`, `Hopper-v1`, `Walker2d-v1`, `Humanoid-v1`  


### Render visualisation of the environment during training

While training, you may start or stop rendering visualisation of the tasks on demand by pressing `r` to *render* or `s` to *stop* rendering. Please keep in mind, this rendering is with an exploratory policy throughout training for BDQ.   


### Saving a model

The current implementation keeps track of the model with the highest average score over the evaluations and saves it to file only at the end of training. 


## Citation

If you find this open-source release useful, please reference in your paper:

```
@inproceedings{tavakoli2018action,
  title={Action Branching Architectures for Deep Reinforcement Learning},
  author={Tavakoli, Arash and Pardo, Fabio and Kormushev, Petar},
  booktitle={AAAI Conference on Artificial Intelligence},
  pages={4131--4138},
  year={2018}
}
```