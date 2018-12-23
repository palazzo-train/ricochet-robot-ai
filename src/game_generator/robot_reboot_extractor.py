import numpy as np

####
#
# extract game state from screen capture of http://www.robotreboot.com
#

class RobotRebootExtractor():
    def __init__(self, im):
        ''' input Image object of the screen capture
        '''

        self.im = im


        self.cell_size = 40
        self.cell_count = 16
        self.goal_check2 = ( 7, -7)

        self.robot_colors = [
                (255,0,0), #r
                (252,255,31), #y
                (0,128,0), #g
                (85,85,255) #b
                ]

        self.goal_colors = [
                # center , check point 2
                [ (255,192,203) , (255,0,0) ] , #r
                [ (252,255,31) , (255,140,0) ] , #y
                [ (144,238,144) , (0,142,0) ] , #g
                [ (0,0,139) , (85,85,255) ]
                ]

        self.robots = [ 0,0,0,0] 
        self.goal = ( [0,0],'x')

        # chip on the right
        self.v_chip = np.zeros( (16,16) , dtype =bool)

        # chip on the bottom
        self.h_chip = np.zeros( (16,16) , dtype =bool)

        self.understand_the_pic()

    def understand_the_pic(self):
        self.offset_x = self.get_x_offset()
        self.offset_y = self.get_y_offset()

        self.get_object_locations()
        self.get_chips()

    def get_chips(self):
        def get_wall( irow, icol ):
            west_x = icol * self.cell_size + self.offset_x
            east_x = west_x + self.cell_size 

            north_y = irow * self.cell_size + self.offset_y
            south_y = north_y + self.cell_size

            #### check v chip on the right
            mid_y = ( south_y + north_y ) /2 

            #print(( irow, icol , east_x , mid_y , west_x , self.offset_x))
            p_right1 = self.im.getpixel( (east_x -1, mid_y))
            p_right2 = self.im.getpixel( (east_x -2 , mid_y))

            right_wall = False
            if ( p_right1 == (0,0,0) ) & ( p_right2 == (0,0,0) ):
                right_wall = True

            #### check h chip on the bottom
            bottom_wall = False
            mid_x = ( east_x + west_x ) /2
            p_bottom1 = self.im.getpixel( ( mid_x , south_y ) )
            p_bottom2 = self.im.getpixel( ( mid_x , south_y -1) )
            if( p_bottom1 == (0,0,0) ) & ( p_bottom2 == (0,0,0)):
                bottom_wall = True

            return right_wall , bottom_wall , east_x , west_x , north_y , south_y

        for irow in range( self.cell_count ):
            for icol in range( self.cell_count ):
                right_wall , bottom_wall , east_x , west_x , north_y , south_y = get_wall(irow, icol)

                self.v_chip[irow,icol] = right_wall
                self.h_chip[irow,icol] = bottom_wall 


    def get_object_locations(self):
        for irow in range(self.cell_count):
            for icol in range(self.cell_count):
                x = ( self.cell_size / 2) + icol * self.cell_size + self.offset_x
                y = ( self.cell_size / 2) + irow * self.cell_size + self.offset_y

                p = self.im.getpixel( (x,y))

                x2 = x + self.goal_check2[0]
                y2 = y + self.goal_check2[1]

                p2 = self.im.getpixel( (x2,y2))

                if ( p != (219,219,219) ) & ( p != (0,0,0)):
                    self.determine_object(irow, icol, p , p2)

    def determine_object(self, irow, icol, p, p2 ):
        ##
        # check goal
        #
        for i , c in zip( range(4) , [ 'r' , 'y' , 'g' , 'b' ] ):
            if ( p == self.goal_colors[i][0] ) & ( p2 == self.goal_colors[i][1] ):
                self.goal = ( np.array( [irow, icol])  , c)
                break

        ##
        # check robot
        #
        for i in range(4):
            if p == self.robot_colors[i] :
                self.robots[i] = [irow, icol]
                break

    ### find the edge
    def get_x_offset(self):
        for i in range(200):
            p1 = self.im.getpixel( ( i , 200 ) )
            p2 = self.im.getpixel( ( i+1 , 200 ) )
            if ( p1 == (0,0,0) ) & ( p2 == (0,0,0) ):
                return i + 1
                break

    def get_y_offset(self):
        for i in range(200):
            p1 = self.im.getpixel( ( 200 , i) )
            p2 = self.im.getpixel( ( 200 , i+1) )
            if ( p1 == (0,0,0) ) & ( p2 == (0,0,0) ):
                return i + 1
                break


