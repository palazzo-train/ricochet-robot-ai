import collections
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from keras import backend as K
from keras.models import model_from_json


class BaseModel():
    def __init__(self,model_save_path):
        self.model_save_path = model_save_path

    def save_model( self ):
        file_name = self.model_save_path + '/' + self.model_name

        model = self.model 

        model_json_file = file_name + '_model.json'
        model_weight_file = file_name + '_weight.h5'
        # serialize model to JSON
        model_json = model.to_json()
        with open(model_json_file, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_weight_file)
        print("Saved model to disk : " + str(file_name))

    def load_model( self, loaded_model ):
        self.model = loaded_model 
        self._compile_model()

    def load_model_from_file(self):
        file_name = self.model_save_path + '/' + self.model_name
        model_json_file = file_name + '_model.json'
        model_weight_file = file_name + '_weight.h5'

        # load json and create model
        json_file = open(model_json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_weight_file)

        self.load_model( loaded_model )
        print("Loaded model from disk : " + str(file_name))





####################
#
#  simple NN model
#
class DQNModel(BaseModel):
    def __init__(self, action_size , board_size, model_save_path='.'):
        super(DQNModel, self).__init__(model_save_path)

        self.model_name = 'NN_128x16'
        self.learning_rate = 0.001
        self.input_dim = np.prod( board_size )

        self.action_size = action_size
        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        self.model = model
        self._compile_model()
        return model

    def _compile_model(self):
        self.model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))


    def predict(self, state):
        act_values = self.model.predict(state)

        return act_values

    def state_conversion(self,state):
        state = state.reshape( [1, self.input_dim])

        return state

    def fit(self, batch_x, batch_y, epochs=1, verbose=0):
        self.model.fit(batch_x, batch_y, epochs=1, verbose=0)

# Deep Q-learning Agent
class DQNAgent():
    '''
    The agent assumes he is holding '-1'  button,  the env is holding '+1'  when training

    if now when the agent is asked to hold '+1' button, we flip all the state variables from -1 to 1 and 1 to -1
    '''

    def __init__(self, board_size, action_size, my_button_color=-1, model_name=None, continue_model =False):
        self.board_size = board_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.30
        self.epsilon_decay = 0.9995



        if my_button_color == -1 :
            self.button_color_invert = 1 # to multiple the state by this varible. meaning no change
        else:
            self.button_color_invert = -1 # to multiple the state by this varible. meaning -1 to 1 , 1 to -1 


        model_save_path  = '../trained_models/four_a_row'
        models = {
            'NN_128x16' : DQNModel,
        }

        if model_name is None:
            model_name = 'NN_128x16'

        self.model = models[model_name](action_size , self.board_size, model_save_path=model_save_path)

        if continue_model:
            print('agent with button ' + str(my_button_color) + ' is loading model')
            self.model.load_model_from_file()

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _flip_state(self,state):
        board, _ , _ = state 

        if self.button_color_invert == -1:
            board = board * self.button_color_invert 

        return board 

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            return action
        
        #state = ( self.board.copy() , act_row, act_col )
        state = self._flip_state(state)
        state = self.model.state_conversion(state)
        act_values = self.model.predict(state)
        
        action = np.argmax(act_values[0])  # returns action

        return action



    def learn(self, state, action, reward, next_state, done):
        #print('state')
        #print(state)
        #print('next state')
        #print(next_state)
        state = self._flip_state(state)
        state = self.model.state_conversion(state)
        next_state = self._flip_state(next_state)
        next_state = self.model.state_conversion(next_state)

        self._remember(state, action, reward, next_state, done)
        self._replay(done, batch_size = 16)


    def _get_training_x_y( self, state, action, reward, next_state, done ):
        target = reward
        
        if not done:
            predict_next_state_action_values = self.model.predict(next_state)
            
            max_next_state_action_value = np.amax( predict_next_state_action_values[0] )
            
            target = reward + self.gamma * max_next_state_action_value

        target_f = self.model.predict(state)

        target_f[0][action] = target

        return state , target_f

    def _replay(self, done, batch_size):
        memory_size = len( self.memory )
        # if done | (memory_size >= batch_size ):
        if (memory_size >= batch_size ):
            batch_size = min( memory_size , batch_size )

            minibatch = random.sample(self.memory, batch_size)

            train_x = []
            train_y = []
            for state, action, reward, next_state, done in minibatch:
                x, y = self._get_training_x_y( state, action, reward, next_state, done )

                train_x.append( x )
                train_y.append( y )

            batch_x = np.concatenate( train_x )
            batch_y = np.concatenate( train_y )

            #self.model.fit(x, y, epochs=1, verbose=0)
            self.model.fit(batch_x, batch_y, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


    def save_model(self):
        self.model.save_model()



