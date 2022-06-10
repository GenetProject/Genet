import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib
matplotlib.use('agg')
import time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
import environments as envs
from utils import *
from param import *
from load_balance.heuristic_agents import *
from actor_agent import *


def run_test(agent, num_exp=100):

    # set up environment
    env = envs.make(args.env)

    all_total_reward = []

    # run experiment
    for ep in range(num_exp):
        print(ep, "--tested" )

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

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)

    # different agents for different environments
    if args.env == 'load_balance':
        schemes = ['least_work', 'learn']
    else:
        schemes = ['learn']

    # tensorflow session
    sess = tf.Session()

    # store results
    all_performance = {scheme: [] for scheme in schemes}

    # create environment
    env = envs.make(args.env)

    # initialize all agents
    agents = {}
    for scheme in schemes:

        if scheme == 'learn':
            agents[scheme] = ActorAgent(sess)
            # saver for loading trained model
            saver = tf.train.Saver(max_to_keep=args.num_saved_models)
            # initialize parameters
            sess.run(tf.global_variables_initializer())
            # load trained model
            if args.saved_model is not None:
                print(args.saved_model, "--------------args.saved_model")
                saver.restore(sess, args.saved_model)

        elif scheme == 'least_work':
            agents[scheme] = LeastWorkAgent()

        elif scheme == 'shortest_processing_time':
            agents[scheme] = ShortestProcessingTimeAgent()

        else:
            print('invalid scheme', scheme)
            exit(1)

    # store results
    all_performance = {}

    # plot job duration cdf
    fig = plt.figure()
    title = 'average: '

    for scheme in schemes:
        print(scheme, "---scheme")

        all_total_reward = run_test(agents[scheme], num_exp=args.num_ep)

        all_performance[scheme] = all_total_reward

        x, y = compute_CDF(all_total_reward)
        plt.plot(x, y)

        title += ' ' + scheme + ' '
        title += '%.2f' % np.mean(all_total_reward)

    plt.xlabel('Total reward')
    plt.ylabel('CDF')
    plt.title(title)
    plt.legend(schemes)

    fig.savefig(args.result_folder + \
        args.env + '_all_performance.png')
    plt.close(fig)

    # save all job durations
    np.save(args.result_folder + \
        args.env + '_all_performance.npy', \
        all_performance)

    sess.close()


if __name__ == '__main__':
    main()
