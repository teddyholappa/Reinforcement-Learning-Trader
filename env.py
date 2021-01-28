from gym import spaces
import numpy as np

class Environment():

	def __init__(self, df, look_back, num_features):
		self.df = df
		self.look_back = look_back
		self.num_features = num_features
		self.total_days = df.shape[0]
		#-1 sell, 0 hold, 1 buy
		self.action_space = [-1,0,1]
		self.observation_shape = (self.look_back*self.num_features,)
		self.prices = self.get_prices()
		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)
		
		self.state = self.set_state()
		self.start_tick = None
		self.end_tick = None
		self.done = False
		self.curr_tick = None
		self.last_trade_tick = None
		self.last_tick = None
		self.postion = None
		self.position_history = None
		self.reward_sum = None
		self.total_profit = None
		self.history = None

	def set_state(self):
		self.start_tick = 0
		self.end_tick = self.look_back
		info = self.df[:self.look_back]
		return info.to_numpy().flatten()

	'''
	Creates new state, calculates reward for the day, tells if we are done
	'''
	def step(self, action):
		
		done = False
		self.start_tick = self.start_tick + 1
		self.end_tick = self.start_tick + self.look_back
		assert self.end_tick - self.start_tick == self.look_back
		if self.end_tick == self.total_days-1: done = True
		new_state = self.df[self.start_tick:self.end_tick].to_numpy().flatten()
		prev_close = self.prices[self.end_tick - 1]
		curr_close = self.prices[self.end_tick]
		reward = action*(curr_close-prev_close)
		return new_state, reward, done


	'''
	This method will be used to calculate profits and loss
	'''
	def get_prices(self):
		prices = self.df["Close"]
		return prices

	'''
	Each index is a list of HLOCVAC for a certain date
	'''
	def data_setup(self):
		array = self.df.to_numpy()
		return array




