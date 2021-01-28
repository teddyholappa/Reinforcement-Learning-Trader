from env import Environment
import pandas as pd 
import numpy as np 
import pandas_datareader.data as web
import torch.optim as optim
import torch.nn as nn
import torch
import datetime
from agent_trader import Agent
from torch.distributions import Categorical
from matplotlib import pyplot as plt

EPISODES = 1000

def train(pi, optimizer):
	#Inner gradient-ascent loop for REINFORCE ALGO
	gamma = .95
	T = len(pi.rewards)
	rets = np.empty(T, dtype=np.float32)
	future_ret = 0.0
	#compute the returns efficiently
	for t in reversed(range(T)):
		future_ret = pi.rewards[t] + gamma * future_ret
		rets = future_ret
	rets = torch.tensor(rets)
	log_probs = torch.stack(pi.log_probs)
	loss = log_probs*rets
	loss=torch.sum(loss)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss

def main():

	start_date = datetime.datetime(2002, 1,1)
	end_date = datetime.datetime(2020, 1,20)
	stocks = web.DataReader('SPY', 'yahoo', start_date, end_date)

	env = Environment(stocks,10, stocks.shape[1])
	in_dim = env.observation_shape[0]
	out_dim = env.action_space.n
	agent = Agent(in_dim, out_dim)
	agent.reset()
	optimizer = optim.Adam(agent.parameters(), lr=0.05)

	for epi in range(EPISODES):
		state = env.set_state()
		for t in range(env.total_days - 15):
			action = agent.act(state)
			state, reward, done = env.step(action)
			agent.rewards.append(reward)
			if done:
				break
		
		loss = train(agent, optimizer)
		total_reward = sum(agent.rewards)
		reward_records = agent.rewards
		solved = total_reward > 195
		agent.reset() #clear memory after training
		print(f'Episode {epi}, loss: {loss}, total_reward: {total_reward}')

	print("----Training Over----")
	cum_rewards = np.cumsum(reward_records)
	sp = env.prices - env.prices[env.look_back]
	plt.title("Cumulative Profit on Last Episode")
	plt.xlabel("Date")
	plt.ylabel("Price ($)")
	plt.plot(cum_rewards)
	plt.show()


if __name__=='__main__':
	main()


