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
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        
        
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            return action
        
        state = state.reshape( [1, 8])

        act_values = self.model.predict(state)
        
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
        self.replay(done, batch_size = 4)


    def get_training_x_y( self, state, action, reward, next_state, done ):
        target = reward
        
        if not done:
            predict_next_state_action_values = self.model.predict(next_state)
            
            max_next_state_action_value = np.amax( predict_next_state_action_values[0] )
            
            target = reward + self.gamma * max_next_state_action_value

        target_f = self.model.predict(state)

        target_f[0][action] = target

        return state , target_f

    def train_by_one_data( self, state, action, reward, next_state, done ):
        target = reward
        
        if not done:
            predict_next_state_action_values = self.model.predict(next_state)
            
            max_next_state_action_value = np.amax( predict_next_state_action_values[0] )
            
            target = reward + self.gamma * max_next_state_action_value

        target_f = self.model.predict(state)

        target_f[0][action] = target
        print('train')
        print( state , target_f )
        print( state.shape , target_f.shape)
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def replay(self, done, batch_size):
        memory_size = len( self.memory )
        if done | (memory_size >= batch_size ):
            batch_size = min( memory_size , batch_size )

            minibatch = random.sample(self.memory, batch_size)

            for state, action, reward, next_state, done in minibatch:
                self.train_by_one_data( state, action, reward, next_state, done )

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


