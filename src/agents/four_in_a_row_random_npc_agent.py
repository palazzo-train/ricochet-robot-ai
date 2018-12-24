class RandomFIAW():
    def __init__(self, b_height , b_width ):
        self.b_height = b_height
        self.b_width = b_width

    def act(self, state):
        #state = ( self.board.copy() , act_row, act_col )
        action = np.random.choice ( np.where( a[0,:] == 0)[0] )

        return action

