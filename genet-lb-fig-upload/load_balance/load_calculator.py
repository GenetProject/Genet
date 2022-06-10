import numpy as np
from param import *


def compute_poisson_uniform_load(workers):
    # poisson job arrival
    # uniform job size distribution
    avg_job_size = np.mean(
        [args.job_size_min, args.job_size_max])
    
    avg_work_per_time = avg_job_size / float(args.job_interval)
    
    total_service_rate = np.sum([w.service_rate for w in workers])

    load = avg_work_per_time / total_service_rate

    return load

def compute_poisson_pareto_load(workers):
    # poisson job arrival
    # pareto job size distribution
    avg_job_size = args.job_size_pareto_shape * \
                   args.job_size_pareto_scale / \
                   (args.job_size_pareto_shape - 1)
    avg_work_per_time = avg_job_size / float(args.job_interval)

    total_service_rate = np.sum([w.service_rate for w in workers])

    load = avg_work_per_time / total_service_rate

    return load

def compute_load(workers):
    if args.job_distribution == 'uniform':
        load = compute_poisson_uniform_load(workers)
    elif args.job_distribution == 'pareto':
        load = compute_poisson_pareto_load(workers)
    else:
        print('Job distribution', args.job_distribution, 'does not exist')
    return load
