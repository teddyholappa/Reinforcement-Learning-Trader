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


EPISODES = 50 # Number of times to iterate through each trajectory
GAMMA = 0.99   # The discount rate
LOOK_BACK = 10 # Number of lookback days


'''
The basic MDP control loop. Initialize the environment,
and train the agent for a total of EPISODES episodes.

The final plot is a visualization of cummulative profits
over time.
'''

def main():

	start_date = datetime.datetime(2002, 1,1)
	end_date = datetime.datetime(2021, 1,30)
	stocks = web.DataReader('SPY', 'yahoo', start_date, end_date)

	env = Environment(stocks,LOOK_BACK, stocks.shape[1])
	in_dim = env.observation_shape[0]
	out_dim = env.action_space.n
	agent = Agent(in_dim, out_dim)
	agent.reset()
	optimizer = optim.RMSprop(agent.parameters(), lr=0.05)

	for epi in range(EPISODES):
		state = env.set_state()
		for t in range(env.total_days - env.look_back):
			action = agent.act(state)
			state, reward, done = env.step(action)
			agent.rewards.append(reward)
			if done:
				break
		
		loss = agent.fit(optimizer, GAMMA)
		total_reward = sum(agent.rewards)
		reward_records = agent.rewards
		agent.reset() #clear memory after training
		print(f'Episode {epi}, Loss: {loss}, Profit: {total_reward}')

	print("----Training Over----")
	cum_rewards = np.cumsum(reward_records)
	plt.title("Cumulative Profit on Last Episode")
	plt.xlabel("Days of Trading")
	plt.ylabel("Price ($)")
	plt.plot(cum_rewards)
	plt.show()


if __name__=='__main__':
	main()


