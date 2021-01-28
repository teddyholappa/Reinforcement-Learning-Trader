import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Agent(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(Agent, self).__init__()
		layers = [nn.Linear(in_dim, 64),
				  nn.ReLU(),
				  nn.Linear(64,64),
				  nn.ReLU(),
				  nn.Linear(64,32),
				  nn.ReLU(),
				  nn.Linear(32, out_dim),
				  nn.Sigmoid()]
		self.model = nn.Sequential(*layers)
		#self.model.apply(self.init_weights)
		self.reset()
		self.actions = [-1, 0, 1]
		self.train()

	def reset(self):
		self.log_probs = []
		self.rewards = []

	def forward(self, x):
		pdparam = self.model(x)
		return pdparam

	def act(self, state):
		x = torch.from_numpy(state.astype(np.float32))
		pdparam = self.forward(x) #This is a forward pass through the network
		m = Categorical(logits=pdparam) #Probability distribution
		action = m.sample()
		log_prob = -m.log_prob(action)
		self.log_probs.append(log_prob) #Store for training
		return self.actions[action.item()]



