[//]: # (Image References)

[image1]: result/tennis.gif "Trained Agent"
[image2]: result/scores.png "Score by episodes"
[image3]: result/average_scores.png "Average score by episodes"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Learning Algorithm

* To train the model, Deep Deterministic Policy Gradient(DDPG) was used which is a type of actor critic method leveraging advantages of both value based methods and policy based methods.
* The model was trained using 2 multi agents reacting to the environment, each with its own copy of the environment(noted as second version).
* For sampling noise, Ornstein-Uhlenbeck process was used as used in the original paper with random sampling from a normal distribution.
* All hyperparameters were set as described in the original paper.
* Reference: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

### Training Code & Weight

* [Notebook](./Tennis.ipynb)
* [Training agent](./ddpg_agent.py)
* [Deep learning model](./model.py)
* [Trained model weight](./checkpoints)

### Criteria

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


### Approach to the problem

#### STEP 1
I started with the DDPG agent and it's model from project 2, Continuous Control. However, after around 20 episodes, the agent was not learning at all and the average score was flattened to 0.

#### STEP 2
It was interesting to see when random agent was set, the average score was fluctuating and sometimes reaching to more than 0.1. It reminded me the lecture saying that randomness in the initial stage could lead to exploration and introduce a better training. Instead of collecting all episodes, I modified the agent to first collect episodes by random agent until the buffer reaches to the `BATCH_SIZE'. In addition, the agent equally added episodes with positive rewards and negative rewards in the experience replay. The result was promising and the agent was able to achieve average score of 0.2, although it went down again to 0.

#### STEP 3
Based on the previous experiment, some of the configuration was updated.
* Until more positive episodes were collected than the `BATCH_SIZE`, random agent was set to decide the action and no training was done until then.
* Positively rewarded episodes were set to be twice more than negatively rewarded episodes in the experience replay.

As a result, the agent was able to be trained to exceed the criteria. After the average score over 100 consecutive episodes went up than 0.5 at episode No.6640, the average score rarely recorded less than 0.5, and the max average score recorded was **0.883** at episode No.9770.

![Average score by episodes][image3]

### Ideas for Future Work

* It would be beneficial to try other type of model architectures, such as adding more depths/units of layers, or using CNN methods.
* For this project, hyperparameters were set to default parameters suggested by the original paper of DDPG. By rigorous hyperparameter tuning, it will help optimize the hyperparameters.
* [Trust Region Policy Optimization (TRPO), Truncated Natural Policy Gradient (TNPG)](https://arxiv.org/abs/1604.06778) and [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://openreview.net/forum?id=SyZipzbCb) can be implemented which have also demonstrated good performance with continuous control tasks.
* [Monte Carlo Tree Search](./alpha-zero) can also be a good candidate for alternative algorithm.
