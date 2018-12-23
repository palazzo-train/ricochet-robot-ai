from PIL import Image
from agent_dqn import DQNAgent
from robot_reboot_extractor import RobotRebootExtractor
from ricochet_env import RicochetEnv



def prepare_game(img_file='game1.png'):
    im = Image.open( '../games/' + img_file).convert('RGB')
    agent = DQNAgent(8 , 4)
    rre = RobotRebootExtractor(im)
    env = RicochetEnv(rre)
    
    return agent , env
