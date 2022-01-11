
import os
import random
import gym
from gym.core import ActionWrapper
from keras.activations import linear
from keras.backend_config import epsilon
import numpy as np 
from collections import deque
from keras.models import Model,load_model
from keras.layers import Input,Dense
from keras.optimizers import adam_v2, rmsprop_v2 

def modelDoubleDQN(input_shape,action_space):
    X_input = Input(input_shape)
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    layer = Dense(512,input_shape=input_shape,activation="relu",kernel_initializer="he_uniform")(X_input)
    #Hidden Layer
    layer = Dense(256,activation="relu",kernel_initializer="he_uniform")(layer)
    #Hidden Layer
    layer = Dense(64,activation="relu",kernel_initializer="he_uniform")(layer)
    #inOutput layer
    layer = Dense(action_space,activation="linear",kernel_initializer="he_uniform")(layer)

    model = Model(inputs= X_input,outputs = layer, name="cartepole_dqn")
    rmsprop_v2.RMSProp
    #Compiling Model : MSE loss function , Optimizer :RMSprop  , rho :discount factor
    model.compile(loss="mse",optimizer=rmsprop_v2.RMSProp(lr=0.00025,rho=0.95,epsilon=0.01),metrics=["accuracy"])
    #model summary
    model.summary() 
    return model
class DoubleDQNagent :
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES=1000
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 #discount
        self.epsilon = 1.0 #exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000
        #create model
        self.model = modelDQN(input_shape=(self.state_size,),action_space=self.action_size)
        
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        if len(self.memory)>self.train_start :
            if self.epsilon>self.epsilon_min:
                #reducing epsilon
                self.epsilon*=self.epsilon_decay
    def act(self,state):
        if np.random.random() <= self.epsilon:
            #exploration
            return random.randrange(self.action_size)
        else :
            #exploit: using model de predict the best action for the state
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory)< self.train_start:
            return
        #randomly sample minibatch from memory
        minibatch = random.sample(self.memory,min(len(self.memory),self.batch_size))
        #initializing state and next state
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size,self.state_size))
        action, reward, done = [], [], [] 

        #separate states actions and rewards and done from mini batch
        # to do before prediction 
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i]= minibatch[i][3]
            done.append(minibatch[i][4])
        
        #define target :predict the action with current state with compiled model without starting to learn (before model.fit  to get the target with initialized values)
        #batch prediction to save time
        #value of target is the Q-value of current state
        
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        for i in range(self.batch_size):
            if done[i]:
                #Correction of the Q value for the action 
                target[i][action[i]]=reward[i]
            else:
                
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])
             
            #train the neural network
            self.model.fit(state,target,batch_size=self.batch_size,verbose=0)
    def load(self,name):
        self.model = load_model(name)
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            #reshape state for neural network input
            state = np.reshape(state, [1, self.state_size])
            done = False 
            i = 0 
            while not done :
                self.env.render()
                action = self.act(state)
                next_state,reward,done,_=self.env.step(action)
                #reshape next state
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i ==self.env._max_episode_steps-1:
                    reward = reward
                else :
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i+=1
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-ddqn.h5")
                        self.save("cartpole-ddqn.h5")
                        return
                self.replay()
    def test(self):
        self.load("cartpole-ddqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    agent = DoubleDQNagent()
    agent.run()
    #agent.test()
