RESEVOIR = 5  # BB
CUSHION = 10  # BB

class BufferBased():

    def __init__(self, fixed=True):
        self.fixed=fixed

    def select_action(self, buffer_size):
        if buffer_size < RESEVOIR:
            bit_rate = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            bit_rate = 6 - 1
        else:
            bit_rate = (6 - 1) * (buffer_size - RESEVOIR) / float( CUSHION )

        #bit_rate = int( bit_rate )

        return bit_rate