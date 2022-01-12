# CartPole_v1_DDQN_DQN
TensorFlow implementation of a Deep Q Network (**DQN**) and Double Deep Q Network (**DDQN**) solving the problem of balancing a pole on cart.
Environment provided by the OpenAI gym.

## Introduction
This project consists of a pole attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
It's The same environment as CartPole-v0 

### Getting Started

* Download the environment  from openAI gym site


* Place the file in the root folder, and unzip (or decompress) the file.

```
python doube_dqn.py :to run the DDQN Algorithm
python dqn.py :to run the DQN Algorithm
```

### Dependencies
*  collections
*  gym
*  numpy
*  random
*  tensorflow

## Detailed Description
---
### DQN
The DQN is the algorithm that combines Q-learning with neural networks
### DDQN
DDQN tries to eliminate the inherent problem of DQN - **overestimation**. The way it does it is through using a different 
target value than DQN. Namely its the following:

![screen shot 2017-09-19 at 1 31 32 am](https://user-images.githubusercontent.com/18519371/30569222-ee9b217c-9cd9-11e7-8bb1-77ddb85f2f39.png)


* to choose the action we use the **online** network weights
* to evaluate the Q function we use the **target** network weights


## Results and discussion
DDQN seems to find the right solution irrespective of the initialization. 


## Resources and links
* https://arxiv.org/abs/1509.06461 - Original paper from Hado van Hasselt
* https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole/2-double-dqn - Similar algorithm in Keras and same hyperparameters


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
