import numpy as np
from param import *


def generate_jobs(num_stream_jobs):

    # time and job size
    all_t = []
    all_size = []

    # generate streaming sequence
    t = 0
    for _ in range(num_stream_jobs):
        if args.job_distribution == 'uniform':
            size = int(np.random.uniform(
                args.job_size_min, args.job_size_max))
        elif args.job_distribution == 'pareto':
            size = int((np.random.pareto(
                args.job_size_pareto_shape) + 1) * \
                args.job_size_pareto_scale)
        else:
            print('Job distribution', args.job_distribution, 'does not exist')

        if args.cap_job_size:
            size = min(int(args.job_size_max), size)

        t += int(np.random.exponential(args.job_interval))

        all_t.append(t)
        all_size.append(size)

    return all_t, all_size
