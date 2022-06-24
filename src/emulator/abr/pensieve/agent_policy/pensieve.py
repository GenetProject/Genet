import csv
import itertools
import logging
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from pensieve.a3c import A3C, compute_entropy
from pensieve.agent_policy import BaseAgentPolicy, RobustMPC
from pensieve.constants import (ACTOR_LR_RATE, A_DIM, CRITIC_LR_RATE,
                                DEFAULT_QUALITY, M_IN_K, S_INFO, S_LEN,
                                VIDEO_BIT_RATE)
from pensieve.utils import write_json_file


class Pensieve(BaseAgentPolicy):
    """Pensieve Implementation.

    Args
        num_agents(int): number of processes to train pensieve models.
        log_dir(str): path where all log files and model checkpoints will be
            saved to.
        actor_path(None or str): path to a actor checkpoint to be loaded.
        critic_path(None or str): path to a critic checkpoint to be loaded.
        model_save_interval(int): the period of caching model checkpoints.
        batch_size(int): training batch size.
        randomization(str): If '', no domain randomization. All
            environment parameters will leave as default values. If 'udr',
            uniform domain randomization. If 'adr', active domain
            randomization.
    """

    def __init__(self, num_agents, log_dir, actor_path=None,
                 critic_path=None, model_save_interval=100, batch_size=100,
                 randomization='', randomization_interval=1):
        # https://github.com/pytorch/pytorch/issues/3966
        # mp.set_start_method("spawn")
        self.num_agents = num_agents

        self.net = A3C(True, [S_INFO, S_LEN], A_DIM,
                       ACTOR_LR_RATE, CRITIC_LR_RATE)
        # NOTE: this is required for the ``fork`` method to work
        # self.net.actor_network.share_memory()
        # self.net.critic_network.share_memory()

        self.load_models(actor_path, critic_path)

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_save_interval = model_save_interval
        self.epoch = 0  # track how many epochs the models have been trained
        self.batch_size = batch_size
        self.randomization = randomization
        self.randomization_interval = randomization_interval
        self.replay_buffer = ReplayBuffer()

    def train(self, train_envs, val_envs=None, test_envs=None, iters=1e5,
              reference_agent_policy=None, use_replay_buffer=False):
        for net_env in train_envs:
            net_env.reset()
        # inter-process communication queues
        net_params_queues = []
        exp_queues = []
        for i in range(self.num_agents):
            net_params_queues.append(mp.Queue(1))
            exp_queues.append(mp.Queue(1))

        # create a coordinator and multiple agent processes
        # (note: threading is not desirable due to python GIL)
        assert len(net_params_queues) == self.num_agents
        assert len(exp_queues) == self.num_agents

        agents = []
        for i in range(self.num_agents):

            agents.append(mp.Process(target=agent,
                                     args=(i, net_params_queues[i],
                                           exp_queues[i], train_envs,
                                           self.log_dir, self.batch_size,
                                           self.randomization,
                                           self.randomization_interval,
                                           self.num_agents)))
        for i in range(self.num_agents):
            agents[i].start()

        self.central_agent(net_params_queues, exp_queues, iters, train_envs,
                           val_envs, test_envs, use_replay_buffer)

        # wait unit training is done
        for i in range(self.num_agents):
            agents[i].join()

    def select_action(self, state):
        bit_rate, action_prob_vec = self.net.select_action(state)
        return bit_rate, action_prob_vec

    def evaluate(self, net_env, save_dir=None):
        torch.set_num_threads(1)
        net_env.reset()
        results = []
        time_stamp = 0
        bit_rate = DEFAULT_QUALITY
        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            state, reward, end_of_video, info = net_env.step(bit_rate)

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms

            results.append([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate],
                            info['buffer_size'], info['rebuf'],
                            info['video_chunk_size'], info['delay'], reward])

            state = torch.from_numpy(state).type('torch.FloatTensor')
            bit_rate, action_prob_vec = self.net.select_action(state)
            bit_rate = np.argmax(action_prob_vec)
            if end_of_video:
                break
        if save_dir is not None:
            # write to a file for the purpose of multiprocessing
            log_path = os.path.join(save_dir, "log_sim_rl_{}".format(
                net_env.trace_file_name))
            with open(log_path, 'w', 1) as f:
                csv_writer = csv.writer(f, delimiter='\t', lineterminator="\n")
                csv_writer.writerow(['time_stamp', 'bitrate', 'buffer_size',
                                     'rebuffer', 'video chunk_size', 'delay',
                                     'reward'])

                csv_writer.writerows(results)
        return results

    def evaluate_envs(self, net_envs):
        arguments = [(net_env, ) for net_env in net_envs]
        with mp.Pool(processes=8) as pool:
            results = pool.starmap(self.evaluate, arguments)
        return results

    def save_models(self, model_save_path):
        """Save models to a directory."""
        self.net.save_actor_model(os.path.join(model_save_path, "actor.pth"))
        self.net.save_critic_model(os.path.join(model_save_path, "critic.pth"))

    def load_models(self, actor_model_path, critic_model_path):
        """Load models from given paths."""
        if actor_model_path is not None:
            self.net.load_actor_model(actor_model_path)
        if critic_model_path is not None:
            self.net.load_critic_model(critic_model_path)

    def central_agent(self, net_params_queues, exp_queues, iters, train_envs,
                      val_envs, test_envs, use_replay_buffer):
        """Pensieve central agent.

        Collect states, rewards, etc from each agent and train the model.
        """
        torch.set_num_threads(2)

        logging.basicConfig(filename=os.path.join(self.log_dir, 'log_central'),
                            filemode='w', level=logging.INFO)

        assert self.net.is_central
        log_header = ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
                      'rewards_median', 'rewards_95per', 'rewards_max']
        test_log_writer = csv.writer(
            open(os.path.join(self.log_dir, 'log_test'), 'w', 1),
            delimiter='\t', lineterminator='\n')
        test_log_writer.writerow(log_header)

        train_e2e_log_writer = csv.writer(
            open(os.path.join(self.log_dir, 'log_train_e2e'), 'w', 1),
            delimiter='\t', lineterminator='\n')
        train_e2e_log_writer.writerow(log_header)

        val_log_writer = csv.writer(
            open(os.path.join(self.log_dir, 'log_val'), 'w', 1),
            delimiter='\t', lineterminator='\n')
        val_log_writer.writerow(log_header)

        t_start = time.time()
        for epoch in range(int(iters)):
            # synchronize the network parameters of work agent
            actor_net_params = self.net.get_actor_param()
            actor_net_params = [params.detach().cpu().numpy()
                                for params in actor_net_params]

            for i in range(self.num_agents):
                net_params_queues[i].put(actor_net_params)
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            # total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            # actor_gradient_batch = []
            # critic_gradient_batch = []
            for i in range(self.num_agents):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
                entropy = info['entropy']
                # add s a r e into replay buffer and sample data out of buffer
                if use_replay_buffer:
                    for s, a, r, e in zip(s_batch, a_batch, r_batch, entropy):
                        self.replay_buffer.add((s, a, r, e))
                    s_batch, a_batch, r_batch, entropy = self.replay_buffer.sample(
                        self.batch_size)

                self.net.get_network_gradient(
                    s_batch, a_batch, r_batch, terminal=terminal,
                    epoch=self.epoch)
                total_reward += np.sum(r_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(entropy)
            print('central_agent: {}/{}, total epoch trained {}'.format(
                epoch, int(iters), self.epoch))

            # log training information
            self.net.update_network()

            avg_reward = total_reward / total_agents
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: {} Avg_reward: {} Avg_entropy: {}'.format(
                self.epoch, avg_reward, avg_entropy))

            if (self.epoch+1) % self.model_save_interval == 0:
                # Save the neural net parameters to disk.
                print("Train epoch: {}/{}, time use: {}s".format(
                    epoch + 1, iters, time.time() - t_start))
                self.net.save_critic_model(os.path.join(
                    self.log_dir, "critic_ep_{}.pth".format(self.epoch + 1)))
                self.net.save_actor_model(os.path.join(
                    self.log_dir, "actor_ep_{}.pth".format(self.epoch + 1)))

                # tmp_save_dir = os.path.join(self.log_dir, 'test_results')
                if val_envs is not None:
                    val_results = self.evaluate_envs(val_envs)
                    vid_rewards = np.array(
                        [np.sum(np.array(vid_results)[1:, -1])
                         for vid_results in val_results])
                    val_log_writer.writerow([self.epoch + 1,
                                             np.min(vid_rewards),
                                             np.percentile(vid_rewards, 5),
                                             np.mean(vid_rewards),
                                             np.median(vid_rewards),
                                             np.percentile(vid_rewards, 95),
                                             np.max(vid_rewards)])
                if test_envs is not None:
                    test_results = self.evaluate_envs(test_envs)
                    vid_rewards = np.array(
                        [np.sum(np.array(vid_results)[1:, -1])
                         for vid_results in test_results])
                    test_log_writer.writerow([self.epoch + 1,
                                              np.min(vid_rewards),
                                              np.percentile(vid_rewards, 5),
                                              np.mean(vid_rewards),
                                              np.median(vid_rewards),
                                              np.percentile(vid_rewards, 95),
                                              np.max(vid_rewards)])
                t_start = time.time()
                # TODO: process val results and write into log
                # evaluate_envs(net, train_envs)
            self.epoch += 1

        # signal all agents to exit, otherwise they block forever.
        for i in range(self.num_agents):
            net_params_queues[i].put("exit")


def agent(agent_id, net_params_queue, exp_queue, net_envs, summary_dir,
          batch_size, randomization, randomization_interval, num_agents):
    """Pensieve agent.

    Performs inference and collect states, rewards, etc.
    """
    torch.set_num_threads(1)
    epoch = 0
    if 'udr' in randomization:
        os.makedirs(os.path.join(summary_dir, "train_envs"), exist_ok=True)
        for env_idx, net_env in enumerate(net_envs):
            env_log_file = os.path.join(
                summary_dir, "train_envs",
                "env{}_agent{}_epoch{}.json".format(env_idx, agent_id, epoch))
            env_dims = net_env.get_dimension_values()
            env_dims['trace_time'] = net_env.trace_time
            env_dims['trace_bw'] = net_env.trace_bw
            write_json_file(env_log_file, env_dims)
    if randomization == 'even_udr':
        for net_env in net_envs:
            for name, dim in net_env.dimensions.items():
                if dim.min_value != dim.max_value:
                    bounds = np.linspace(dim.min_value, dim.max_value,
                                         num_agents+1)
                    net_env.dimensions[name].min_value = bounds[agent_id]
                    net_env.dimensions[name].max_value = bounds[agent_id+1]
    # set random seed
    prng = np.random.RandomState(agent_id)
    # print(str(net_envs[0].get_dims_with_rand()['buffer_threshold']))

    with open(os.path.join(summary_dir,
                           'log_agent_'+str(agent_id)), 'w', 1) as log_file:
        csv_writer = csv.writer(log_file, delimiter='\t', lineterminator="\n")
        # 'time_stamp', 'bit_rate', 'buffer_size',
        # 'rebuffer', 'video_chunk_size', 'delay','chunk_idx',
        csv_writer.writerow(['epoch', 'avg_chunk_reward',  'trace_name',
                             'video_chunk_length', 'buffer_threshold',
                             'link_rtt', 'drain_buffer_sleep_time',
                             'packet_payload_portion', 'T_l', 'T_s', 'cov',
                             'duration', 'step', 'min_throughput',
                             'max_throughput'])

        # initial synchronization of the network parameters from the
        # coordinator
        net = A3C(False, [S_INFO, S_LEN], A_DIM, ACTOR_LR_RATE, CRITIC_LR_RATE)
        actor_net_params = net_params_queue.get()
        if actor_net_params == "exit":
            return
        net.hard_update_actor_network(actor_net_params)

        time_stamp = 0
        env_idx = prng.randint(len(net_envs))
        net_env = net_envs[env_idx]
        bit_rate = DEFAULT_QUALITY
        s_batch = []
        a_batch = []
        r_batch = []
        video_chunk_rewards = []
        entropy_record = []
        is_1st_step = True
        epoch_randomization = 0  # track in which epoch randomization occurs
        while True:

            # the action is from the last decision
            # this is to make the framework similar to the real
            state, reward, end_of_video, info = net_env.step(bit_rate)

            bit_rate, action_prob_vec = net.select_action(state)
            bit_rate = bit_rate.item()
            # Note: we need to discretize the probability into 1/RAND_RANGE
            # steps, because there is an intrinsic discrepancy in passing
            # single state and batch states

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms
            if not is_1st_step:
                s_batch.append(state)
                a_batch.append(bit_rate)
                r_batch.append(reward)
                video_chunk_rewards.append(reward)
                entropy_record.append(compute_entropy(action_prob_vec)[0])
            else:
                # ignore the first chunck since we can't control it
                is_1st_step = False

            # log time_stamp, bit_rate, buffer_size, reward
            env_params = net_env.get_dimension_values()
            if len(s_batch) == batch_size:
                exp_queue.put([np.concatenate(s_batch), np.array(a_batch),
                               np.array(r_batch), end_of_video,
                               {'entropy': np.array(entropy_record)}])

                actor_net_params = net_params_queue.get()
                if actor_net_params == "exit":
                    break
                net.hard_update_actor_network(actor_net_params)
                s_batch = []
                a_batch = []
                r_batch = []
                entropy_record = []
                epoch += 1
            if end_of_video:
                # time_stamp, VIDEO_BIT_RATE[bit_rate],
                #                      info['buffer_size'], info['rebuf'],
                #                      info['video_chunk_size'], info['delay'],
                # net_env.nb_chunk_sent,
                csv_writer.writerow([epoch,
                                     np.mean(np.array(video_chunk_rewards)),
                                     net_env.trace_file_name,
                                     env_params['video_chunk_length'],
                                     env_params['buffer_threshold'],
                                     env_params['link_rtt'],
                                     env_params['drain_buffer_sleep_time'],
                                     env_params['packet_payload_portion'],
                                     env_params['T_l'],
                                     env_params['T_s'],
                                     env_params['cov'],
                                     env_params['duration'],
                                     env_params['step'],
                                     env_params['min_throughput'],
                                     env_params['max_throughput']])
                net_env.reset(random_start=True)
                if randomization == '':
                    pass  # no randomization
                elif 'udr' in randomization:
                    if epoch - epoch_randomization >= randomization_interval:
                        for tmp_env in net_envs:
                            tmp_env.randomize(None)
                        epoch_randomization = epoch
                        os.makedirs(os.path.join(summary_dir, "train_envs"),
                                    exist_ok=True)
                        for env_idx, net_env in enumerate(net_envs):
                            env_log_file = os.path.join(
                                summary_dir, "train_envs",
                                "env{}_agent{}_epoch{}.json".format(
                                    env_idx, agent_id, epoch))
                            env_dims = net_env.get_dimension_values()
                            env_dims['trace_time'] = net_env.trace_time
                            env_dims['trace_bw'] = net_env.trace_bw
                            write_json_file(env_log_file, env_dims)
                else:
                    raise NotImplementedError
                env_idx = prng.randint(len(net_envs))
                net_env = net_envs[env_idx]
                bit_rate = DEFAULT_QUALITY
                time_stamp = 0
                is_1st_step = True
                video_chunk_rewards = []


class ReplayBuffer(object):
    """Simple replay buffer."""

    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.next_idx = 0

    def add(self, data: Tuple):
        """Add tuples of (state, action, reward)."""
        assert len(data) == 4
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size: int = 100):
        """Randomly sample batch_size of (state, action, reward)."""
        print("sample", len(self.storage))
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, rewards, entropies = [], [], [], []

        for i in ind:
            state, action, reward, entropy = self.storage[i]
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            entropies.append(np.array(entropy, copy=False))

        return np.array(states), np.array(actions), np.array(rewards), np.array(entropies)


def compare_mpc_pensieve(pensieve_abr, val_envs, param_ranges):
    num_small_ranges = 3
    mpc_abr = RobustMPC()
    candidates = []
    range_indices = []
    for param in sorted(param_ranges):
        boundaries = np.linspace(*param_ranges[param], num_small_ranges+1)
        # need a seed here
        values = [np.random.uniform(lower, upper)
                  for lower, upper in zip(boundaries[:-1], boundaries[1:])]
        candidates.append(values)
        range_indices.append(list(range(num_small_ranges)))

    mpc_pensieve_reward_diffs = []
    for param_tuple in itertools.product(*candidates):
        randomized_values = {}
        for param, value in zip(sorted(param_ranges), param_tuple):
            randomized_values[param] = value
        print(param_tuple, randomized_values)
        for net_env in val_envs:
            net_env.randomize(randomized_values)
        mpc_results = mpc_abr.evaluate_envs(val_envs)
        pensieve_results = pensieve_abr.evaluate_envs(val_envs)
        mpc_reward = np.mean(np.concatenate(
            [np.array(vid_results)[1:, -2] for vid_results in mpc_results]))
        pensieve_reward = np.mean(np.concatenate(
            [np.array(vid_results)[1:, -1]
                for vid_results in pensieve_results]))
        mpc_pensieve_reward_diffs.append(mpc_reward - pensieve_reward)
    print(mpc_pensieve_reward_diffs)
    max_gap_range_idx = np.argmax(mpc_pensieve_reward_diffs)
    final_range_idices = list(itertools.product(
        *range_indices))[max_gap_range_idx]
    print(mpc_pensieve_reward_diffs[max_gap_range_idx], final_range_idices)

    selected_ranges = {}
    for param, range_idx in zip(sorted(param_ranges), final_range_idices):
        boundaries = np.linspace(*param_ranges[param], num_small_ranges+1)
        selected_ranges[param] = (boundaries[:-1][range_idx],
                                  boundaries[1:][range_idx])
    print(selected_ranges)
    return selected_ranges
