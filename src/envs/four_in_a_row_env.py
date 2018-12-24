import gym
import numpy as np
from gym import spaces
import itertools as it

class RandomNpcAgent():
    def __init__(self, b_height , b_width ):
        self.b_height = b_height
        self.b_width = b_width

    def act(self, state):
        #state = ( self.board.copy() , act_row, act_col )
        board , _ , _ = state
        action = np.random.choice ( np.where( board[0,:] == 0)[0] )

        return action

class FourInARowEnv(gym.Env):

    '''

        board = 6 x7 
        bottom = -1 (red) , 1 (green) , 0 (nothing) 
        player is red


        state = ( self.board.copy() , act_row, act_col )
    '''


    def __init__(self, npc_agent=None):
        self.b_width = 7
        self.b_height = 6
        self.in_row_count = 4


        self.action_space = spaces.Discrete( 7 )
        self.observation_space = None

        self.player_button = -1
        self.npc_button = 1

        if npc_agent is None:
            self.npc_agent = RandomNpcAgent( self.b_height , self.b_width)
        else:
            self.npc_agent = npc_agent

        ####
        #
        # render paramter
        #
        self.cell_size = 40

        self.reset()

    def step(self, action):

        act_col= action
        done = False

        done, act_row , act_col = self.player_step(act_col, self.player_button )

        # player won
        if done:
            reward = 1
            state = ( self.board.copy() , act_row, act_col )
            return state , reward , done, ''

        # NPC response
        npc_action = self.npc_agent.act( (self.board, -1 , -1))
        done , _ , _  = self.player_step(npc_action, self.npc_button )

        #check is npc won
        if done:
            reward = -1
            state = ( self.board.copy() , act_row, act_col )
            return state , reward , done, ''

        reward = 0
        state = ( self.board.copy() , act_row, act_col )

        return state , reward , done, ''


    def reset(self):
        self.board = np.zeros( (self.b_height , self.b_width)).astype(int)
        self.avail_row = np.ones( self.b_width ) .astype(int) * ( self.b_height - 1)
        self.t = 0

        return self.board.copy() , -1 , -1

    ################
    #
    #  board setup
    #
    def init_board(self, rrg):
        self.cell_count = rrg.cell_count
        self.init_robots = np.array(rrg.robots)
        self.goal = rrg.goal

        self.v_chip = rrg.v_chip
        self.h_chip = rrg.h_chip
        self.done = False

    ################
    #
    #  state transition
    #
    def player_step(self, act_col, act_button):
        done = False
        # the column is full
        if self.avail_row[act_col] == -1 :
            act_row = -1
            return done, act_row , act_col 

        self.board[ self.avail_row[act_col] , act_col ] = act_button
        act_row = self.avail_row[act_col] 
        self.avail_row[act_col] -= 1



        done = self.check_win( act_button , act_row , act_col)
        return done, act_row , act_col

    def check_win(self, act_button, act_row , act_col):
        # check horizonal
        for i in range( 0 , self.in_row_count):
            if np.sum( self.board[ act_row, i:i+self.in_row_count]) == (act_button * self.in_row_count ):
                return True

        # check vertical 
        for i in range( 0 , self.in_row_count):
            if np.sum( self.board[ i:i+self.in_row_count, act_col]) == (act_button * self.in_row_count ):
                return True

        # check diagonal
        # from bottom left to top right
        for i in range( self.in_row_count -1 , -1 , -1):
            row0 = act_row + i 
            col0 = act_col - i

            row1 = row0 - self.in_row_count -1 
            col1 = col0 + self.in_row_count -1 

            if ( row1 >= 0 ) & (row0 < self.b_height) & (col0 >= 0) & (col1 < self.b_width ):
                total_b = 0
                for j in range(self.in_row_count):
                    irow = row0 - j
                    icol = col0 + j

                    total_b += self.board[ irow , icol ]

                if total_b == (act_button * self.in_row_count):
                    return True
                
        # check diagonal
        # from top left to bottom right
        for i in range( self.in_row_count -1 , -1 , -1):
            row0 = act_row - i 
            col0 = act_col - i

            row1 = row0 + self.in_row_count -1 
            col1 = col0 + self.in_row_count -1 

            if ( row0 >= 0 ) & (row1 < self.b_height) & (col0 >= 0) & (col1 < self.b_width ):
                total_b = 0
                for j in range(self.in_row_count):
                    irow = row0 + j
                    icol = col0 + j

                    total_b += self.board[ irow , icol ]

                if total_b == (act_button * self.in_row_count):
                    return True

        return False


    #####################################
    #
    #  render
    #
    def render(self, mode):

        if mode == 'rgb_array' :
            buffer = np.ones( ( self.cell_count * self.cell_size , self.cell_count * self.cell_size , 3) ).astype('uint8') * 255

            self.draw_grid(buffer)
            self.draw_chip(buffer)
            self.draw_robot(buffer)
            self.draw_goal(buffer)
            return buffer

        raise NotImplementedError

    def draw_goal(self, buffer):
        s = 12
        robot_colors = [
                (255,0,0), #r
                (252,255,31), #y
                (0,128,0), #g
                (85,85,255) #b
                ]
        (irow , icol ), c= self.goal

        x = icol * 40 
        y = irow * 40

        robot_i = self.color_list.index( c )

        c = robot_colors[robot_i]

        for i , j in it.product( range( s) , repeat = 2 ):
            buffer[ y + i , x + j ] = c
            buffer[ y + 40 - i , x + 40 - j ] = c
            buffer[ y + 40 - i , x + j ] = c
            buffer[ y + i , x + 40 - j ] = c

    def draw_circle(self, buffer , c, mid_x , mid_y ):
        s = 12 

        for i , j in it.product( range( -s , +s ) , repeat =2 ) :
            buffer[ mid_y + i , mid_x + j ] = c

    def draw_robot(self,buffer):
        robot_colors = [
                (255,0,0), #r
                (252,255,31), #y
                (0,128,0), #g
                (85,85,255) #b
                ]

        for i , c in zip( range(4) , robot_colors) :
            irow, icol = self.robots[i]

            mid_x = icol * 40 + 20
            mid_y = irow * 40 + 20

            self.draw_circle( buffer , c, mid_x , mid_y )

    def draw_chip(self,buffer):
        for irow , icol in it.product( range(self.cell_count) , repeat = 2):
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
        for irow , icol in it.product( range(self.cell_count) , repeat = 2):
            start_x = icol * self.cell_size
            start_y = irow * self.cell_size

            for x in range( start_x , start_x + self.cell_size ):
                buffer[ x, start_y ] = [ 0,0,0]
                buffer[ x, start_y + self.cell_size - 1] = [ 0,0,0]

            for y in range( start_y , start_y + self.cell_size ):
                buffer[ start_x, y ] = [ 0,0,0]
                buffer[ start_x + self.cell_size - 1 , y] = [ 0,0,0]
