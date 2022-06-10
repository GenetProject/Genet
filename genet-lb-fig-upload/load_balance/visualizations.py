import matplotlib
matplotlib.use('agg')
import heapq
import numpy as np
import matplotlib.pyplot as plt
from param import *
from load_calculator import *


def visualize_queue_occupancy(workers, finished_jobs,
                              file_name='queue_occupancy_visualization'):

    # overall job occupancy
    overall_time, overall_jobs = [], []
    # job occupacny for each worker
    worker_time = [[] for _ in xrange(args.num_workers)]
    worker_jobs = [[] for _ in xrange(args.num_workers)]
    
    # all job arrival/departure events
    events = []

    # all finished jobs
    for job in finished_jobs:
        # True for arrival, False for completion
        heapq.heappush(events, (job.arrival_time, (True, job.worker.worker_id)))
        heapq.heappush(events, (job.finish_time, (False, job.worker.worker_id)))

    # all waiting jobs (if any)
    for worker in workers:
        for job in worker.queue:
            # True for arrival
            heapq.heappush(events, (job.arrival_time, (True, worker.worker_id)))
            assert job.finish_time is None  # job not yet started

    # all running jobs (if any)
        for worker in workers:
            if worker.curr_job is not None:
                heapq.heappush(events, \
                    (worker.curr_job.arrival_time, (True, worker.worker_id)))
                assert worker.curr_job.finish_time >= worker.wall_time.curr_time

    curr_overall_jobs = 0
    curr_worker_jobs = [0] * args.num_workers

    while len(events) > 0:
        (t, (arrival_event, worker_id)) = heapq.heappop(events)
        
        overall_time.append(t)
        worker_time[worker_id].append(t)

        if arrival_event:
            curr_overall_jobs += 1
            curr_worker_jobs[worker_id] += 1

        else:
            curr_overall_jobs -= 1
            assert curr_overall_jobs >= 0
            curr_worker_jobs[worker_id] -= 1
            assert curr_worker_jobs[worker_id] >= 0

        overall_jobs.append(curr_overall_jobs)
        worker_jobs[worker_id].append(curr_worker_jobs[worker_id])

    # compute analytical load
    load = compute_load(workers)

    # compute average job completion time
    job_completion_time = []
    worker_job_completion_time = [[] for _ in xrange(args.num_workers)]

    for job in finished_jobs:
        duration = job.finish_time - job.arrival_time
        job_completion_time.append(duration)
        worker_job_completion_time[job.worker.worker_id].append(duration)

    fig = plt.figure()

    ax = plt.subplot(args.num_workers + 1, 1, 1)
    ax.step(overall_time, overall_jobs, 'blue', alpha=0.85, where='post')
    ax.set_title('Load: ' + '%.2f' % load + ' Total service rate: ' + \
                 '%.2f' % sum(w.service_rate for w in workers) + \
                 ' Average completon time: ' + \
                 '%.2f' % np.mean(job_completion_time))
    ax.set_xlabel('Time')
    ax.set_ylabel('Queue size')

    for worker_id in xrange(args.num_workers):
        ax = plt.subplot(args.num_workers + 1, 1, worker_id + 2)
        ax.step(worker_time[worker_id], worker_jobs[worker_id],
                'blue', alpha=0.85, where='post')
        ax.set_title('Server '+ str(worker_id) + \
                     ' service rate: ' + \
                     '%.2f' % workers[worker_id].service_rate + \
                     ' Average completion time: ' + \
                     '%.2f' % np.mean(worker_job_completion_time[worker_id]))
        ax.set_xlabel('Time')
        ax.set_ylabel('Queue size')

    plt.tight_layout()

    plt.savefig(args.result_folder + file_name + '.png')
    plt.close(fig)
