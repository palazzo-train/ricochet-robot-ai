import gym
import numpy as np
from gym import spaces
import itertools as it

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

    def step(self, action):
        robot_id , dir_id = self.decode_action( action )

        cur_row , cur_col = self.robots[robot_id]

        ## 0 -> N , 1 -> E , 2 -> S , 3 -> W
        act_func = [ self.go_N , self.go_E , self.go_S , self.go_W ]

        new_row , new_col = act_func[dir_id]( robot_id , cur_row , cur_col )

        self.robots[ robot_id ] = [ new_row , new_col ]
        reward = -1

        # check done
        target_robot = ['r' , 'y' , 'g' , 'b' ].index( self.goal[2] )
        target_loc = [ self.goal[0] , self.goal[1] ]

        done = False
        if self.robots[target_robot] == target_loc :
            done = True

        return self.robots , reward , done, ''

    def reset(self):
        self.robots = self.init_robots.copy()
        self.done = False

        return self.robots

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

    def go_E( self, robot_id, cur_row , cur_col ):
        if cur_col == self.cell_count -1 :
            return cur_row , cur_col 
        else:
            for icol in range( cur_col , self.cell_count ):
                if icol == self.cell_count -1:
                    return cur_row ,icol 

                # check chip
                if self.v_chip[cur_row , icol ]:
                    return cur_row , icol

                # check robot
                for j in range(4):
                    if j == robot_id:
                        continue

                    # has robot 
                    if self.robots[j] == [ cur_row , icol + 1] :
                        return cur_row , icol

    def go_W( self, robot_id, cur_row , cur_col ):
        if cur_col == 0:
            return cur_row , cur_col 
        else:
            for icol in range( cur_col , -1 , -1 ):
                if icol == 0:
                    return cur_row , icol

                # check chip
                if self.v_chip[cur_row , icol -1]:
                    return cur_row , icol

                # check robot
                for j in range(4):
                    if j == robot_id:
                        continue

                    # has robot  
                    if self.robots[j] == [ cur_row , icol - 1] :
                        return cur_row , icol

    
    def go_S( self, robot_id, cur_row , cur_col ):
        if cur_row == self.cell_count -1:
            return cur_row , cur_col 
        else:
            for irow in range( cur_row , self.cell_count):
                if irow == self.cell_count -1:
                    return irow , cur_col

                # check chip
                if self.h_chip[irow , cur_col]:
                    return irow , cur_col

                # check robot
                for j in range(4):
                    if j == robot_id:
                        continue

                    # has robot  
                    if self.robots[j] == [ irow + 1, cur_col] :
                        return irow, cur_col


    def go_N( self, robot_id, cur_row , cur_col ):
        if cur_row == 0:
            return cur_row , cur_col 
        else:
            for irow in range( cur_row , -1,-1):
                if irow == 0:
                    return irow , cur_col

                # check chip
                if self.h_chip[irow -1, cur_col]:
                    return irow , cur_col

                # check robot
                for j in range(4):
                    if j == robot_id:
                        continue

                    # has robot  
                    if self.robots[j] == [ irow - 1, cur_col] :
                        return irow, cur_col

    #####################################
    #
    #  render
    #
    def render(self):
        buffer = np.ones( ( self.cell_count * self.cell_size , self.cell_count * self.cell_size , 3) ).astype('uint8') * 255

        self.draw_grid(buffer)
        self.draw_chip(buffer)
        self.draw_robot(buffer)
        self.draw_goal(buffer)


        return buffer

    def draw_goal(self, buffer):
        s = 12
        robot_colors = [
                (255,0,0), #r
                (252,255,31), #y
                (0,128,0), #g
                (85,85,255) #b
                ]
        irow , icol , c= self.goal

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
