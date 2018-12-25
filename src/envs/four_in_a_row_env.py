import gym
import numpy as np
from gym import spaces
import itertools as it
import random
from four_dqn_agent import DQNAgent

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

        if npc_agent is not None:
            self.npc_agent = npc_agent
        else:
            self.npc_agent = RandomNpcAgent( self.b_height , self.b_width)

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

        # wrong move
        # state no change
        if act_row == -1: # wrong move
            reward = -10
            done = True
            state = ( self.board.copy() , act_row, act_col )
            return state , reward , done, ''

        # player won
        if done:
            reward = 1
            state = ( self.board.copy() , act_row, act_col )
            return state , reward , done, ''

        self.placed_button += 1

        ## check no space
        if self.placed_button >= self.max_placed_button :
            # no space
            reward = 0
            done = True
            state = ( self.board.copy() , act_row, act_col )
            return state , reward , done, ''

        # NPC response
        npc_action = self.npc_agent.act( (self.board, -1 , -1))
        done , act_row , _  = self.player_step(npc_action, self.npc_button )
        # wrong move by npc
        # state no change
        if act_row == -1: # wrong move
            # the npc agen made wrong move. fallback to a random and move again
            npc_action = np.random.choice ( np.where( self.board[0,:] == 0)[0] )

            done , act_row , _  = self.player_step(npc_action, self.npc_button )

        #check is npc won
        if done:
            reward = -1
            state = ( self.board.copy() , act_row, act_col )
            return state , reward , done, ''

        reward = 0
        state = ( self.board.copy() , act_row, act_col )

        self.placed_button += 1

        ## check no space
        ##
        ##  draw game
        if self.placed_button >= self.max_placed_button :
            # no space
            reward = 0
            done = True
            state = ( self.board.copy() , act_row, act_col )
            return state , reward , done, ''


        return state , reward , done, ''


    def reset(self):
        self.board = np.zeros( (self.b_height , self.b_width)).astype(int)
        self.avail_row = np.ones( self.b_width ) .astype(int) * ( self.b_height - 1)
        self.t = 0
        self.placed_button = 0
        self.max_placed_button = (self.b_height * self.b_width)

        act_row = -1
        act_col = -1

        ### random, sometime npc moves first
        if np.random.rand() >= 0.25 :
            npc_action = self.npc_agent.act( (self.board, -1 , -1))
            done, act_row , act_col = self.player_step(npc_action, self.npc_button)
            state = ( self.board.copy() , act_row, act_col )
            return state 


        state = ( self.board.copy() , act_row, act_col )
        return state


    def manual_step(self, action , player):
        reward = 0
        done , act_row, act_col = self.player_step(action , player )
        state = ( self.board.copy() , act_row, act_col )
        return state , reward , done, ''



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
            act_row = -1 # wrong move
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
            buffer = np.ones( ( self.b_height* self.cell_size , self.b_width* self.cell_size , 3) ).astype('uint8') * 255

            self.draw_grid(buffer)
            self.draw_button(buffer)
            #self.draw_chip(buffer)
            #self.draw_robot(buffer)
            #self.draw_goal(buffer)
            return buffer

        raise NotImplementedError


    def draw_button(self, buffer):
        for irow , icol in it.product(  range(self.b_height) , range( self.b_width)  ):
            start_x = icol * self.cell_size
            start_y = irow * self.cell_size

            if self.board[irow, icol] == self.player_button:
                for x , y in it.product( range( start_x + 4, start_x + self.cell_size -4 ) , range(start_y +4 , start_y + self.cell_size -4 ) ):
                    buffer[ y, x] = [ 255,0,0]

            elif self.board[irow, icol] == self.npc_button:
                for x , y in it.product( range( start_x + 4, start_x + self.cell_size -4 ) , range(start_y + 4, start_y + self.cell_size -4 ) ):
                    buffer[ y, x] = [ 0,255,0]



    def draw_grid(self, buffer):
        for irow , icol in it.product(  range(self.b_height) , range( self.b_width)  ):
            start_x = icol * self.cell_size
            start_y = irow * self.cell_size

            for x in range( start_x , start_x + self.cell_size ):
                buffer[ start_y , x] = [ 0,0,0]
                buffer[ start_y + self.cell_size - 1 , x] = [ 0,0,0]

            for y in range( start_y , start_y + self.cell_size ):
                buffer[ y , start_x ] = [ 0,0,0]
                buffer[ y, start_x + self.cell_size - 1 ] = [ 0,0,0]
