import gym
import numpy as np
from gym import spaces

class RicochetEnv(gym.Env):

    '''

        board = 16 x 16
        robot = r , y , g , b

        robots location : ndarray (4,2)

        r = robots[0]
        y = robots[1]
        g = robots[2]
        b = robots[3]


        goal = ( irow , icol , color )

        vertical and horizonal chips maps

        v_chip[6,2] => there is a vertical chip on the right wall of 6,2
        h_chip[6,2] => there is a horizonal chip on the bottom wall of 6,2

    '''


    def __init__(self, robot_reboot_generator):
        self.cell_count = 16
        self.color_list = [ 'r' , 'y' , 'g' , 'b' ]
        self.dir_list = [ 'N' , 'E' , 'S' , 'W' ]

        self.init_board(robot_reboot_generator)

        self.action_space = spaces.Discrete( 4*4 )
        self.observation_space = None


        ####
        #
        # render paramter
        #
        self.cell_size = 40

        self.reset()

    def step(self, ation):
        None

    def reset(self):
        self.robots = self.init_robots.copy()
        self.done = False

    ################
    #
    #  board setup
    #
    def init_board(self, rrg):
        self.cell_count = rrg.cell_count
        self.init_robots = rrg.robots
        self.goal = rrg.goal

        self.v_chip = rrg.v_chip
        self.h_chip = rrg.h_chip
        self.done = False

    ################
    #
    #  state transition
    #
    def encode_action( self, robot_color , dir_code ):

        robot_i = self.color_list.index( robot_color )
        dir_i = self.dir_list.index( dir_code )

        action = robot_i * 4 + dir_i  

        return action

    def decode_action( self, action) :
        robot_i = int( action / 4)
        dir_i = action % 4

        return robot_i , dir_i


    #####################################
    #
    #  render
    #
    def render(self):
        buffer = np.ones( ( self.cell_count * self.cell_size , self.cell_count * self.cell_size , 3) ).astype('uint8') * 255

        self.draw_grid(buffer)
        self.draw_chip(buffer)


        return buffer


    def draw_chip(self,buffer):
        for irow in range(self.cell_count):
            for icol in range(self.cell_count ):

                if self.v_chip[irow, icol] :
                    if icol != self.cell_count -1:
                        start_x = icol * 40
                        start_y = irow * 40

                        for y in range( start_y , start_y + self.cell_size ):
                            for i in range(5):
                                buffer[ y , start_x + self.cell_size - i  ] = [ 90,90,90]

                if self.h_chip[irow, icol]:
                    if irow != self.cell_count -1:
                        start_x = icol * 40
                        start_y = irow * 40

                        for x in range( start_x , start_x + self.cell_size ):
                            for i in range(5):
                                #print( irow, icol , [ start_y + self.cell_size - i -1 , x ] )
                                buffer[ start_y + self.cell_size - i -1 , x  ] = [ 90,90,90]

    def draw_grid(self, buffer):
        for irow in range(self.cell_count):
            for icol in range(self.cell_count):
                start_x = icol * self.cell_size
                start_y = irow * self.cell_size

                for x in range( start_x , start_x + self.cell_size ):
                    buffer[ x, start_y ] = [ 0,0,0]
                    buffer[ x, start_y + self.cell_size - 1] = [ 0,0,0]

                for y in range( start_y , start_y + self.cell_size ):
                    buffer[ start_x, y ] = [ 0,0,0]
                    buffer[ start_x + self.cell_size - 1 , y] = [ 0,0,0]
