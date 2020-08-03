[//]: # (Image References)

[image1]: result/reacher.gif "Trained Agent"
[image2]: result/wo_adding_Grad.png "Without addiing torch.no_grad Option for Critic"
[image3]: result/adding_Grad.png "After addiing torch.no_grad Option for Critic"


# Project 2: Continuous Control

### Introduction

For this project, the model was trained with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Learning Algorithm

* To train the model, Deep Deterministic Policy Gradient(DDPG) was used which is a type of actor critic method leveraging advantages of both value based methods and policy based methods.
* The model was trained using 20 identical agents, each with its own copy of the environment(noted as second version).
* For sampling noise, Ornstein-Uhlenbeck process was used as used in the original paper with random sampling from a normal distribution.
* All hyperparameters were set as described in the original paper.
* Reference: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

### Training Code & Weight

* [Notebook](./Continuous_Control.ipynb)
* [Training agent](./ddpg_agent.py)
* [Deep learning model](./model.py)
* [Trained model weight](./result)

### Criteria

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
* This yields an average score for each episode (where the average is over all 20 agents).


By using second version, the model was able to be trained in multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience. When `torch.no_grad` option was not specified for the `critic` while `actor` was training, the reward was able to go over 12 and did not maintain for 100 consecutive episodes.

![Without addiing torch.no_grad Option for Critic][image2]

After adding `torch.no_grad` option to `critic` while `actor` was training, the model was able to achieve more than average score of 30 after 88 episodes, and was stabilize for the rest of the training at average score of 34.

![After addiing torch.no_grad Option for Critic][image3]

### Ideas for Future Work

* It would be beneficial to try other type of model architectures, such as adding more depths/units of layers, or using CNN methods.
* For this project, hyperparameters were set to default parameters suggested by the original paper of DDPG. By rigorous hyperparameter tuning, it will help optimize the hyperparameters.
* [Trust Region Policy Optimization (TRPO), Truncated Natural Policy Gradient (TNPG)](https://arxiv.org/abs/1604.06778) and [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://openreview.net/forum?id=SyZipzbCb) can be implemented which have also demonstrated good performance with continuous control tasks.