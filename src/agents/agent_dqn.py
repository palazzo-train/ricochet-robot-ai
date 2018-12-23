import collections
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random


# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        print('act =============  ' + str(self.epsilon))
        
        
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            print(action)
            return action
        
        print(' act state')
        #print( state )
        state = state.reshape( [1, 8])
        act_values = self.model.predict(state)
        
        #print('aaa')
        print(act_values)
        action = np.argmax(act_values[0])  # returns action
        return action


    def learn(self, state, action, reward, next_state, done):
        #print('state')
        #print(state)
        #print('next state')
        #print(next_state)
        state = state.reshape( [1, 8])
        next_state = next_state.reshape( [1, 8])
        self.remember(state, action, reward, next_state, done)
        self.replay(batch_size = 16)

    def replay(self, batch_size):
        if len( self.memory ) < batch_size :
            return 

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            next_state = next_state.reshape( [1, 8])
            print('target: ' + str(target))
            if not done:
                print('kkkkkkkkkkkkkkk')
                print('state')
                print(state)
                print('predicct')
                
                
                print('new next state')
                print(next_state)
                
                predict_next_state_action_values = self.model.predict(next_state)
                
                max_next_state_action_value = np.amax( predict_next_state_action_values[0] )
                
                print(predict_next_state_action_values)
                print(max_next_state_action_value)
                target = reward + self.gamma * max_next_state_action_value

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


