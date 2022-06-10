import numpy as np

class Job(object):
    def __init__(self, size, arrival_time):
        self.size = size
        self.arrival_time = arrival_time
        self.worker = None
        self.start_time = None
        self.finish_time = None
