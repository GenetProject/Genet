import random

def conditional_fy(l, p):
    """Shuffle elements of a list with a given probability

    Args:
        l: list
        p: shuffle probability
            (0: elements are never shuffled,
             1: elements are always shuffled)

    """
    #print("-------prob p")
    assert 0 <= p <= 1

    for i in range(len(l) - 1, 0, -1):
        shuffle = random.random()
        if shuffle < p:
            j = random.randint(0, i - 1)
            l[i], l[j] = l[j], l[i]
    return l

class Worker(object):
    def __init__(self, worker_id, service_rate, wall_time, queue_shuffle_prob):
        self.worker_id = worker_id
        self.service_rate = service_rate
        self.wall_time = wall_time
        self.queue_shuffle_prob = queue_shuffle_prob
        self.queue = []
        self.curr_job = None

    def schedule(self, job):
        self.queue.append(job)
        job.worker = self

    def process(self):
        # if the worker is currently idle (no current
        # job or current job is done), and there are jobs
        # in the queue, then FIFO process a job
        if (self.curr_job is None or \
           self.curr_job.finish_time <= self.wall_time.curr_time) \
           and len(self.queue) > 0:
            self.queue = conditional_fy(self.queue, p=self.queue_shuffle_prob)
            self.curr_job = self.queue.pop(0)
            duration = int(self.curr_job.size / self.service_rate)
            self.curr_job.start_time = self.wall_time.curr_time
            self.curr_job.finish_time = self.wall_time.curr_time + duration

            return self.curr_job

        else:
            return None

    def reset(self):
        self.queue = []
        self.curr_job = None
