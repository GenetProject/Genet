import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib
matplotlib.use('agg')
import time
import multiprocessing as mp
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
import environments as envs
from utils import *
from param import *
from load_balance_actor_agent import *
from critic_agent import *
from average_reward import *
from tensorboard_summaries import *
from actor_critic_test import run_test
from load_balance.heuristic_agents import *

SCALE=1e9

def run_test(agent, num_exp=50):
    # set up environment
    env = envs.make(args.env)

    all_total_reward = []

    # run experiment
    for ep in range(num_exp):
        # print(ep, "------ep")
        env.set_random_seed(100000000 + ep)
        env.reset()

        total_reward = 0

        state = env.observe()
        done = False

        while not done:
            act = agent.get_action(state)
            state, reward, done = env.step(act)
            total_reward += reward

        all_total_reward.append(total_reward)

    return all_total_reward


def main():
    if args.param_name == "SERVICE_RATES":
        args.service_rates = [args.current_value/3, args.current_value, args.current_value*3]
    if args.param_name == "JOB_SIZE_MAX":
        args.job_size_max = args.current_value
    if args.param_name == "JOB_INTERVAL":
        args.job_interval = args.current_value
    if args.param_name == "NUM_JOBS":
        args.num_jobs = args.current_value
    if args.param_name == "QUEUE_SHUFFLE_PROB":
        args.queue_shuffle_prob = args.current_value


    sess = tf.Session()

    actor_agent = ActorAgent( sess )

    # initialize parameters
    sess.run( tf.global_variables_initializer() )
    saver = tf.train.Saver()

    # load trained model
    if args.saved_model is not None:
        #print(args.saved_model, "------args.saved_model")
        saver.restore( sess ,args.saved_model )


    test_result_rl = run_test(actor_agent)
    test_mean = np.mean( test_result_rl )/ SCALE
    test_std = np.std( test_result_rl )/ SCALE
    print([test_mean, test_std],  "------rl mean, std")

    sess.close()


if __name__ == '__main__':
    main()
