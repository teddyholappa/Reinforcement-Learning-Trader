import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

'''
A basic agent fit with a NN to estimate the objective function,
an act method that determines whether to buy or sell the stock,
and fit function which trains the NN after each episode.
'''

class Agent(nn.Module):

	def __init__(self, in_dim, out_dim):

		super(Agent, self).__init__()
		layers = [nn.Linear(in_dim, 64),
				  nn.ReLU(),
				  nn.Linear(64,32),
				  nn.ReLU(),
				  nn.Linear(32,16),
				  nn.ReLU(),
				  nn.Linear(16, out_dim),
				  nn.Sigmoid()]
		self.model = nn.Sequential(*layers)
		self.actions = [-1, 1] #-1 is sell, 1 is buy
		self.reset()
		self.train()
		
	def reset(self):

		self.log_probs = []
		self.rewards = []

	def act(self, state):

		x = torch.from_numpy(state).type(torch.FloatTensor)
		estimate = self.model(x) #This is a forward pass through the network
		prob_dist = Categorical(logits=estimate) #Probability distribution for sampling our action
		action = prob_dist.sample()
		log_prob = prob_dist.log_prob(action)
		
		#Store for training, think of this as policy_history
		self.log_probs.append(log_prob) 
		return self.actions[action.item()]

	def fit(self, optimizer, gamma):

		l = len(self.rewards)
		discounted_profits = np.empty(l,dtype=np.float32)
		discounted_return = 0.

		for t in reversed(range(l)):
			discounted_return = self.rewards[t] + discounted_return*gamma
			discounted_profits[t] = discounted_return

		discounted_profits = torch.tensor(discounted_profits)
		log_probs = torch.stack(self.log_probs)
		loss = -log_probs*discounted_profits
		loss = torch.sum(loss)
		optimizer.zero_grad()
		loss.backward() #Backpropagation
		optimizer.step() #Gradient ascent - Max the reward function!
		return loss



