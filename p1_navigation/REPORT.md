[//]: # (Image References)

[image1]: result/banana.gif "Trained Agent"
[image2]: result/DQN.png "Result of Deep Q-Network"
[image3]: result/DQN-wDropoutLayers.png "Result of Deep Q-Network after adding dropout layers"
[image4]: result/DDQN.png "Result of Double Deep Q-Network"
[image5]: result/DDQN-wDropoutLayers.png "Result of Deep Q-Network after adding dropout layers"
[image6]: result/DDQN-wDropoutLayers-PPER.png "Result of Deep Q-Network after adding dropout layers and proportional prioritized experience replay"


# Project 2: Continuous Control

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Learning Algorithm

* To train the model, Double Deep Q-Network(DQN) with Proportional Prioritized Experience Replay was used which is a value based methods.
* Dropout layers were added to the model, and had seen improvement with the score.
* Proportional Prioritized Experience Replay was used with stratified sampling as described in the original paper.
* Reference:
  - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
  - [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

### Training Code & Weight

* [Notebook](./Navigation.ipynb)
* [Training agent](./dqn_agent.py)
* [Deep learning model](./model.py)
* [Trained model weight](./model)

### Criteria

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

5 cases were tested to compare different model architectures.

#### STEP 1
![Result of Deep Q-Network][image2]

Deep Q-Network algorithm was applied to train the model and exceeded the criteria requirements. Although the average score was +13 over 100 consecutive episodes, scores oscillated in a wide amplitude from almost 0 to 25.

#### STEP 2
![Result of Deep Q-Network after adding dropout layers][image3]

After adding dropout layers to the model, training speed became slower as I introduced more variance to the model. However, training became more stabilied and average score was slightly improved to +15.

#### STEP 3
![Result of Double Deep Q-Network][image4]

Changing the algorithm to DDQN from DQN did not provide a big lift.

#### STEP 4
![Result of Deep Q-Network after adding dropout layers][image5]

After adding dropout layers to DDQN, the average score went +15 over 100 consecutive episodes, which was a big improvement from DQN with dropout layers.

#### STEP 5
![Result of Deep Q-Network after adding dropout layers and proportional prioritized experience replay][image6]

Lastly, proportional prioritized experience replay was added to the DDQN agent with dropout layers. The average max score reached to 16.7 which was the highest socre from multiple variations.

### Ideas for Future Work

* It would be beneficial to try other type of model architectures, such as adding more depths/units of layers, or using CNN methods with the image environment.
* For this project, hyperparameter tuning was not done and the steps focused more on applying methods described in [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298). If the hyperparameters are more optimized, it would lead not only to better score, but also faster training.