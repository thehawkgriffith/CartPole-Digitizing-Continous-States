import gym 
import os
import sys
from datetime import datetime
import numpy as np
from gym import wrappers


def generate_state(features):
	for index in range(len(features)):
		features[index] = str(int(features[index]))
	state = int("".join(features))
	return state

def to_bin(value, bins):
	bin = np.digitize([value], bins)[0]
	return bin


class FeatureTransformer():

	def __init__(self):
		self.cart_pos = np.linspace(-2.4, 2.4, 9)
		self.cart_vel = np.linspace(-2, 2, 9)
		self.pole_ang = np.linspace(-0.4, 0.4, 9)
		self.pole_vel = np.linspace(-3.5, 3.5, 9)

	def transform(self, state):
		c_pos, c_vel, p_ang, p_vel = state
		c_pos = to_bin(c_pos, self.cart_pos)
		c_vel = to_bin(c_vel, self.cart_vel)
		p_ang = to_bin(p_ang, self.pole_ang)
		p_vel = to_bin(p_vel, self.pole_vel)
		features = [c_pos, c_vel, p_ang, p_vel]
		state = generate_state(features)
		return state


class Model():

	def __init__(self, env, feature_transformer):
		self.env = env
		self.feature_transformer = feature_transformer
		num_states = 10000
		num_actions = 2
		self.Q = np.random.uniform(-1, 1, (num_states, num_actions))

	def predict(self, state):
	    state = self.feature_transformer.transform(state)
	    return self.Q[state]

	def update(self, action, G, state):
		state = self.feature_transformer.transform(state)
		self.Q[state, action] += 1e-2*(G - self.Q[state, action])

	def sample_action(self, state, eps):
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			prediction = self.predict(state)
			return np.argmax(prediction)


def play_an_episode(model, eps, gamma):
	state = env.reset()
	done = False
	total_reward = 0
	iters = 0
	while not done and iters < 10000:
		action = model.sample_action(state, eps)
		prev_state = state
		state, reward, done, _ = env.step(action)
		total_reward += reward
		if done and iters < 199:
			reward = -300
		G = reward + gamma * np.max(model.predict(state))
		model.update(action, G, prev_state)
		iters += 1
	return total_reward

env = gym.make('CartPole-v0')
ft = FeatureTransformer()
model = Model(env, ft)
gamma = 0.9
if 'monitor' in sys.argv:
	filename = os.path.basename(__file__).split('.')[0]
	monitor_dir = './' + filename + '_' + str(datetime.now())
	env = wrappers.Monitor(env, monitor_dir)

N = 1000
totalrewards = np.empty(N)
for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play_an_episode(model, eps, gamma)
    totalrewards[n] = totalreward
    if n % 100 == 0:
        print("Episode: ", n, "Total Reward Attained: ", totalreward, "Epsilon:", eps)
print("Average Reward for the last 100 Episodes:", totalrewards[-100:].mean())
print("Total Steps:", totalrewards.sum())



