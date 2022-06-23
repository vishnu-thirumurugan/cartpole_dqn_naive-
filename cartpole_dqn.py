# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 08:44:35 2022

@author: oe21s024

"""
'''
 this file contains the deep q network class, agent class
 this is a good practice for objcet oriented programming 
 this is just a naive implementation, it does not fit for very situation
'''
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from util import plot_learning_curve

class LinearDeepQNetwork(nn.Module): # in nn init part itself we declare loss function and optimizers
    
    def __init__(self, lr, n_actions, input_dims): # lr is for optimizer
        super(LinearDeepQNetwork, self).__init__() # super constructor
        
        
        # neural network architecture
        self.fc1 = nn.Linear(*input_dims, 128) # *input_dims is used to unpack the list and to freely use any numbers
        self.fc2 = nn.Linear(128, n_actions)

        # optimizer, loss, device, sending data to device 
        self.optimizer = optim.Adam(self.parameters(), lr =lr)
        self.loss = nn.MSELoss() 
        self.device = T.device('cuda : 0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward (self, state):
        
        layer1 = F.relu(self.fc1(state)) # we use F to make use of activation function
        actions = self.fc2(layer1) # not activated as it is the output ---> q value of different actions ---> we need that raw q value 
        
        return actions
    
    
class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma = 0.99, eps_start = 1.00, eps_decay = 0.00001, eps_end = 0.01):
        
        self.lr = lr
        self.gamma = gamma 
        
        self.n_actions = n_actions
        self.input_dims = input_dims 
        
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        self.action_space = [i for i in range(self.n_actions)]
        
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        
    def choose_action (self, observation):
       # for choosing action we use epsilon greedy
       # if random number is greater than epsilon, we choose max q value
       # for this we need to call linear q network to find max q network 
       # for this network, state is the input 
       if np.random.random() > self.epsilon :
           state   = T.tensor(observation, dtype = T.float).to(self.Q.device)
           actions = self.Q.forward(state)
           action  = T.argmax(actions).item()
           
       else:
           action = np.random.choice(self.action_space)
    
       return action
   
    
   
    
   
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_decay \
                                if self.epsilon > self.eps_end else self.eps_end
                                    
    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad() # never forget to set optimizer to zero grad for each episode 
        states = T.tensor(state, dtype = T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        reward = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_ , dtype = T.float).to(self.Q.device)
        
        
        q_pred = self.Q.forward(states)[actions] # max q value from nn
        q_next = self.Q.forward(states_).max()
        q_target = reward + self.gamma * q_next
        
        loss = self.Q.loss(q_pred, q_target).to(self.Q.device)
        loss.backward()   # back propogation
        self.Q.optimizer.step()  # step function in environment of gym
        self.decrement_epsilon()
        

if __name__ == '__main__' :
    env = gym.make('CartPole-v1')
    scores = []
    eps_history = []
    n_games = 10000
    
    
    agent = Agent(lr = 0.001, input_dims = env.observation_space.shape,
                  n_actions = env.action_space.n)
    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()
        
        
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward # adding reward for one complete episode
            agent.learn(obs, action, reward, obs_)
            obs = obs_ # updating current state 
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        
        if  i % 100 == 0 :
            avg_score = np.mean(scores[-100:])
            print('episode', i, 'score %.1f avg score %.1f epsilon %.2f' % (score, avg_score, agent.epsilon))
        
        # plotting
    
    filename = ('cartpole_naive_dqn.png')
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
        
            
            
    