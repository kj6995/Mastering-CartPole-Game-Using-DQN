import gym
import csv
import random
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.models import load_model

class Memory:
    def __init__(self, Memorysize):
        self.Memorysize = Memorysize
        self.inputcount = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.nextState = []
        self.isFinal = []

    def getMiniBatch(self, Memorysize) :
        indices = random.sample(np.arange(len(self.states)), min(Memorysize,len(self.states)) )
        miniBatch = []
        for index in indices:
            miniBatch.append({'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.nextState[index], 'isFinal': self.isFinal[index]})
        return miniBatch

    def getMemorySize(self) :
        return len(self.states)

    def getState(self, index): 
        return {'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.nextState[index], 'isFinal': self.isFinal[index]}

    def addState(self, state, action, reward, newState, isFinal) :
        if (self.inputcount >= self.Memorysize - 1) :
            self.inputcount = 0
        if (len(self.states) > self.Memorysize) :
            self.states[self.inputcount] = state
            self.actions[self.inputcount] = action
            self.rewards[self.inputcount] = reward
            self.nextState[self.inputcount] = newState
            self.isFinal[self.inputcount] = isFinal
        else :
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.nextState.append(newState)
            self.isFinal.append(isFinal)
        
        self.inputcount += 1

class DeepQ:
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, beginStart):
        self.size_input = inputs
        self.size_output = outputs
        self.memory = Memory(memorySize)
        self.discountFactor = discountFactor
        self.beginStart = beginStart
        self.learningRate = learningRate

    def initializeNetworks(self, hiddenLayers):
        model = self.createModel(self.size_input, self.size_output, hiddenLayers, "relu", self.learningRate)
        self.model = model

        targetModel = self.createModel(self.size_input, self.size_output, hiddenLayers, "relu", self.learningRate)
        self.targetModel = targetModel

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.size_output, input_shape=(self.size_input,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.size_input,), init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            model.add(Dense(self.size_output, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print "layer ",i,": ",weights
            i += 1


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    def calculateTargetDouble(self,qValuesNewState, reward, isFinal,qValuesNewStateNew):
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * qValuesNewState[np.argmax(qValuesNewStateNew)]

    

    def calculateTarget(self, qValuesNewState, reward, isFinal):
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.size_output)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def addState(self, state, action, reward, newState, isFinal):
        self.memory.addState(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getMemorySize() >= 1:
            return self.memory.getState(self.memory.getMemorySize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):      
        if self.memory.getMemorySize() > self.beginStart:
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.size_input), dtype = np.float64)
            Y_batch = np.empty((0,self.size_output), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.size_output]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)



env = gym.make('CartPole-v0')
epochs = 200
steps = 10000
updateTargetNetwork = 10000
explorationRate = 1
minibatch_size = 128
beginStart = 128
learningRate = 0.00025
discountFactor = 0.99
memorySize = 1000000

last100Scores = [0] * 100
last100ScoresIndex = 0
last100Filled = False

deepQ = DeepQ(4, 2, memorySize, discountFactor, learningRate, beginStart)

deepQ.initializeNetworks([300,300])


stepCounter = 0
exploration_list = []
episode_score = []

for epoch in xrange(epochs):
    exploration_list.append([explorationRate,explorationRate])    
    observation = env.reset()
    print explorationRate

    for t in xrange(steps):
        env.render()
        qValues = deepQ.getQValues(observation)

        action = deepQ.selectAction(qValues, explorationRate)

        newObservation, reward, done, info = env.step(action)

        if (t >= 199):
            print "reached the end! :D"
            done = True
            if(abs(newObservation[1]) < 0.5 and newObservation[3] < 0.5):
                reward = 200            
            else:
                reward = -100

        if done and t < 199:
            print "decrease reward"
            reward = -200

        deepQ.addState(observation, action, reward, newObservation, done)

        if stepCounter >= beginStart:
            if stepCounter <= updateTargetNetwork:
                deepQ.learnOnMiniBatch(minibatch_size, False)
            else :
                deepQ.learnOnMiniBatch(minibatch_size, True)

        observation = newObservation

        if done:
            last100Scores[last100ScoresIndex] = t
            last100ScoresIndex += 1
            if last100ScoresIndex >= 100:
                last100Filled = True
                last100ScoresIndex = 0
            if not last100Filled:
                print "Episode ",epoch," finished after {} timesteps".format(t+1)
            else :
                print "Episode ",epoch," finished after {} timesteps".format(t+1)," last 100 average: ",(sum(last100Scores)/len(last100Scores))
            break

        stepCounter += 1
        if stepCounter % updateTargetNetwork == 0:
            deepQ.updateTargetNetwork()
            print "updating target network"

    episode_score.append([t,t])
    explorationRate *= 0.995

    explorationRate = max (0.05, explorationRate)
    # if(epoch%5 == 4):
    #     deepQ.model.save('models/model'+str(epoch)+'.h5')


writer1 = csv.writer(open("epsilon.csv", 'w'))
writer2 = csv.writer(open("episodeScore.csv", 'w'))
for row1 in exploration_list:
    writer1.writerow(row1)
for row2 in episode_score:
    writer2.writerow(row2)
