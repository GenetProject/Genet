import numpy as np


class LeastWorkAgent(object):
    def __init__(self):
        pass

    def get_action(self, state):
        workers, _, _ = state

        min_work_idx = None
        min_work = np.inf

        for i in range(len(workers)):
            worker = workers[i]
            work = np.sum([j.size for j in worker.queue])
            if work < min_work:
                min_work_idx = i
                min_work = work

        return min_work_idx

class ShortestProcessingTimeAgent(object):
    def __init__(self):
        pass

    def get_action(self, state):
        workers, _, _ = state

        min_time_idx = None
        min_time = np.inf

        for i in range(len(workers)):
            worker = workers[i]
            work = np.sum([j.size for j in worker.queue])
            remain_time = work / worker.service_rate
            if remain_time < min_time:
                min_time_idx = i
                min_time = remain_time

        return min_time_idx
