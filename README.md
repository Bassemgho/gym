# CartPole_v1_DDQN_DQN
TensorFlow implementation of a Deep Q Network (**DQN**) and Double Deep Q Network (**DDQN**) solving the problem of balancing a pole on cart.
Environment provided by the OpenAI gym.

## Introduction
This project consists of a pole attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
It's The same environment as CartPole-v0 

### Getting Started

* Download the environment from one of the links below. You need only select the environment that matches your operating system:

Linux
Mac OSX:
Windows (32-bit) 
Windows (64-bit)
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the environment.

* Place the file in the root folder, and unzip (or decompress) the file.

```
python doube_dqn.py
python dqn.py
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
DDQN seems to find the right solution irrespective of the initialization. See below an evolution of the score for one run:
![screen shot 2017-09-19 at 1 43 23 am](https://user-images.githubusercontent.com/18519371/30569493-84f81002-9cdb-11e7-9d3a-e699c351f912.png)


## Resources and links
* https://arxiv.org/abs/1509.06461 - Original paper from Hado van Hasselt
* https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole/2-double-dqn - Similar algorithm in Keras and same hyperparameters


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
