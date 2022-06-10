import argparse

parser = argparse.ArgumentParser(description='parameters')

# -- Basic --
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='epsilon (default: 1e-6)')
parser.add_argument('--result_folder', type=str, default='./results/',
                    help='Result folder path (default: ./results)')
parser.add_argument('--model_folder', type=str, default='./models/',
                    help='Model folder path (default: ./models)')

# -- Learning --
parser.add_argument('--input_dim', type=int, default=13,
                    help='input dimension (default: 13)')
parser.add_argument('--hid_dims', type=int, default=[200, 128], nargs='+',
                    help='hidden dimensions (default: [200, 128])')
parser.add_argument('--action_dim', type=int, default=6,
                    help='action dimension (default: 6)')
parser.add_argument('--num_agents', type=int, default=10,
                    help='number of training agents (default: 10)')
parser.add_argument('--saved_model', type=str, default=None,
                    help='path to the saved model (default: None)')
parser.add_argument('--num_saved_models', type=int, default=1000,
                    help='Number of models to keep (default: 1000)')
parser.add_argument('--model_save_interval', type=int, default=200,
                    help='Interval for saving Tensorflow model (default: 100)')
parser.add_argument('--num_models', type=int, default=10,
                    help='Number of models for value network (default: 10)')
parser.add_argument('--num_ep', type=int, default=100,
                    help='Number of training epochs (default: 10000)')
parser.add_argument('--lr_rate', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--gamma', type=float, default=1,
                    help='discount factor (default: 1)')
parser.add_argument('--reward_scale', type=float, default=1e4,
                    help='reward scale in training (default: 1e4)')
parser.add_argument('--diff_reward', type=int, default=0,
                    help='Differential reward mode (default: 0)')
parser.add_argument('--clip_grads', type=int, default=0,
                    help='clip gradients (default: 0)')
parser.add_argument('--meta_inner_step', type=int, default=1,
                    help='meta learning inner training step (default: 1)')
parser.add_argument('--average_reward_storage', type=int, default=100000,
                    help='Storage size for average reward (default: 100000)')
parser.add_argument('--stop_grad', type=int, default=1,
                    help='Ignore higher order gradients (default: 1)')
parser.add_argument('--entropy_weight_init', type=float, default=1,
                    help='Initial exploration entropy weight (default: 1)')
parser.add_argument('--entropy_weight_min', type=float, default=0.0001,
                    help='Final minimum entropy weight (default: 0.0001)')
parser.add_argument('--entropy_weight_decay', type=float, default=1e-3,
                    help='Entropy weight decay rate (default: 1e-3)')
parser.add_argument('--reset_prob', type=float, default=1e-5,
                    help='Probability for episode to reset (default: 1e-5)')
parser.add_argument('--reset_prob_decay', type=float, default=4.5e-9,
                    help='Decay rate of reset probability (default: 4.5e-9)')
parser.add_argument('--reset_prob_min', type=float, default=1e-6,
                    help='Minimum of decay probability (default: 1e-6)')
parser.add_argument('--num_stream_jobs_grow', type=float, default=4.5,
                    help='growth rate of number of streaming jobs  (default: 4.5)')
parser.add_argument('--num_stream_jobs_max', type=float, default=10000,
                    help='maximum number of number of streaming jobs (default: 10000)')

# -- Environment --
parser.add_argument('--env', type=str, default='load_balance',
                    help='name of environment (default: load_balance)')

# -- Load Balance --
parser.add_argument('--num_workers', type=int, default=3,
                    help='number of workers (default: 3)')
parser.add_argument('--num_stream_jobs', type=int, default=2000,
                    help='number of streaming jobs (default: 2000)')
parser.add_argument('--service_rates', type=float, default=[0.5, 1.0, 2.0], nargs='+',
                    help='workers service rates (default: [0.5, 1.0, 2.0])')
parser.add_argument('--service_rate_min', type=float, default=1.0,
                    help='minimum service rate (default: 1.0)')
parser.add_argument('--service_rate_max', type=float, default=10.0,
                    help='maximum service rate (default: 4.0)')
parser.add_argument('--job_distribution', type=str, default='uniform',
                    help='Job size distribution (default: uniform)')
parser.add_argument('--job_size_min', type=float, default=10.0,
                    help='minimum job size (default: 100.0)')
parser.add_argument('--job_size_max', type=float, default=10000.0,
                    help='maximum job size (default: 10000.0)')
parser.add_argument('--job_size_norm_factor', type=float, default=1000.0,
                    help='normalize job size in the feature (default: 1000.0)')
parser.add_argument('--job_size_pareto_shape', type=float, default=2.0,
                    help='pareto job size distribution shape (default: 2.0)')
parser.add_argument('--job_size_pareto_scale', type=float, default=100.0,
                    help='pareto job size distribution scale (default: 100.0)')
parser.add_argument('--job_interval', type=int, default=100,
                    help='job arrival interval (default: 100)')
parser.add_argument('--cap_job_size', type=int, default=0,
                    help='cap job size below max (default: 0)')
parser.add_argument('--queue_shuffle_prob', type=float, default=0.5,
                    help='queue shuffle prob (default: 0.5)')

# -- BO --
parser.add_argument( '--param_name', type=str, default='MAX_THROUGHPUT', help='param BO is testing on' )
parser.add_argument( '--current_value', type=float, default='10', help='BO input number' )

# -- UDR --
parser.add_argument( '--param_service_rates_low', type=float, default='0.1')
parser.add_argument( '--param_service_rates_high', type=float, default='10')

parser.add_argument( '--param_job_size_max_low', type=int, default='100')
parser.add_argument( '--param_job_size_max_high', type=int, default='10000')

parser.add_argument( '--param_job_interval_low', type=int, default='10')
parser.add_argument( '--param_job_interval_high', type=int, default='1000')

parser.add_argument( '--param_num_stream_jobs_low', type=int, default='10')
parser.add_argument( '--param_num_stream_jobs_high', type=int, default='10000')

parser.add_argument( '--param_queue_shuffle_prob_low', type=float, default='0.1')
parser.add_argument( '--param_queue_shuffle_prob_high', type=float, default='1')

args = parser.parse_args()