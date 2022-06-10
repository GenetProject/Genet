# import gym
from load_balance.env import LoadBalanceEnvironment


def make(env_name):
    if env_name == 'load_balance':
        env = LoadBalanceEnvironment()

    else:
        print('Environment ' + env_name + ' is not supported')
        exit(1)

    return env