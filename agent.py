import gym
import random
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.models import load_model


model = load_model('modelsDouble/model240.h5')


env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env,'.',force = True)

t = 0
observation = env.reset()
while(True):
	env.render()
	qValues = model.predict(observation.reshape(1,len(observation)))
	action = np.argmax(qValues)
	observation, reward, done, info = env.step(action)
	t = t + 1
	if(done):
		print(t)
		break


