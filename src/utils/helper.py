from PIL import Image
from agent_dqn import DQNAgent
from robot_reboot_extractor import RobotRebootExtractor
from ricochet_env import RicochetEnv
from keras import backend as K
import numpy as np

from keras.models import model_from_json


def prepare_game(img_file='game1.png'):
    im = Image.open( '../games/' + img_file).convert('RGB')
    agent = DQNAgent(8 , 4*4)
    rre = RobotRebootExtractor(im)
    env = RicochetEnv(rre)
    
    return agent , env


def save_model(model, file_name='model'):
    model_json_file = file_name + '_model.json'
    model_weight_file = file_name + '_weight.h5'
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weight_file)
    print("Saved model to disk")
    

def load_model(file_name='model'):
    model_json_file = file_name + '_model.json'
    model_weight_file = file_name + '_weight.h5'
    # load json and create model
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weight_file)
    print("Loaded model from disk")


    return loaded_model


def show_trainable_params(model):

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))

