import numpy as np
from utils import *
from param import *
from load_balance.worker import *
from load_balance.job import *
from load_balance.timeline import *
from load_balance.wall_time import *
from load_balance.job_generator import *


class LoadBalanceEnvironment(object):
    def __init__(self):
        # global timer
        self.wall_time = WallTime()
        # uses priority queue
        self.timeline = Timeline()
        # total number of streaming jobs (can be very large)
        self.num_stream_jobs = args.num_stream_jobs
        # workers
        self.workers = self.initialize_workers(args.service_rates)
        # episode retry probability
        self.reset_prob = 0
        # current incoming job to schedule
        self.incoming_job = None
        # finished jobs
        self.finished_jobs = []

    def generate_jobs(self):
        all_t, all_size = generate_jobs(self.num_stream_jobs)
        for t, size in zip(all_t, all_size):
            self.timeline.push(t, size)

    def initialize(self):
        assert self.wall_time.curr_time == 0
        new_time, obj = self.timeline.pop()
        self.wall_time.update(new_time)
        assert isinstance(obj, int)  # a job arrival event
        size = obj
        self.incoming_job = Job(size, self.wall_time.curr_time)

    def initialize_workers(self, service_rates):
        workers = []

        for worker_id in range(args.num_workers):
            if service_rates is None:
                service_rate = np.random.uniform(
                    args.service_rate_min, args.service_rate_max)
            else:
                service_rate = service_rates[worker_id]
            worker = Worker(worker_id, service_rate, self.wall_time, args.queue_shuffle_prob)
            workers.append(worker)

        return workers

    def observe(self):
        return self.workers, self.incoming_job, self.wall_time.curr_time

    def reset(self):
        for worker in self.workers:
            worker.reset()
        self.wall_time.reset()
        self.timeline.reset()
        self.generate_jobs()
        self.max_time = generate_coin_flips(self.reset_prob)
        self.incoming_job = None
        self.finished_jobs = []
        # initialize environment (jump to first job arrival event)
        self.initialize()

    def set_random_seed(self, seed):
        np.random.seed(seed)

    def step(self, action):

        # schedule job to worker
        self.workers[action].schedule(self.incoming_job)
        running_job = self.workers[action].process()
        if running_job is not None:
            self.timeline.push(running_job.finish_time, running_job)

        # erase incoming job
        self.incoming_job = None

        # set to compute reward from this time point
        reward = 0

        while len(self.timeline) > 0:

            new_time, obj = self.timeline.pop()

            # update reward
            num_active_jobs = sum(len(w.queue) for w in self.workers)
            for worker in self.workers:
                if worker.curr_job is not None:
                    assert worker.curr_job.finish_time >= \
                           self.wall_time.curr_time  # curr job should be valid
                    num_active_jobs += 1
            reward -= (new_time - self.wall_time.curr_time) * num_active_jobs

            # tick time
            self.wall_time.update(new_time)

            if new_time >= self.max_time:
                break

            if isinstance(obj, int):  # new job arrives
                size = obj
                self.incoming_job = Job(size, self.wall_time.curr_time)
                # break to consult agent
                break

            elif isinstance(obj, Job):  # job completion on worker
                job = obj
                self.finished_jobs.append(job)
                if job.worker.curr_job == job:
                    # worker's current job is done
                    job.worker.curr_job = None
                running_job = job.worker.process()
                if running_job is not None:
                    self.timeline.push(running_job.finish_time, running_job)

            else:
                print("illegal event type")
                exit(1)

        done = ((len(self.timeline) == 0) and \
               self.incoming_job is None) or \
               (self.wall_time.curr_time >= self.max_time)

        return self.observe(), reward, done
