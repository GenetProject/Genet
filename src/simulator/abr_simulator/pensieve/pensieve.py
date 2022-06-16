import csv
import logging
import multiprocessing as mp
import os
import time
from typing import List

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# import visdom


from simulator.abr_simulator.abr_trace import AbrTrace
from simulator.abr_simulator.base_abr import BaseAbr
from simulator.abr_simulator.schedulers import TestScheduler
from simulator.abr_simulator.constants import (
    A_DIM,
    BUFFER_NORM_FACTOR,
    CRITIC_LR_RATE,
    DEFAULT_QUALITY,
    S_INFO,
    S_LEN,
    VIDEO_BIT_RATE,
    M_IN_K,
    VIDEO_CHUNK_LEN,
    MILLISECONDS_IN_SECOND,
    TOTAL_VIDEO_CHUNK,
    TRAIN_SEQ_LEN
)
from simulator.abr_simulator.env import Environment
from simulator.abr_simulator.pensieve import a3c
from simulator.abr_simulator.utils import plot_abr_log, linear_reward


BITRATE_DIM = 6
# CHUNK_TIL_VIDEO_END_CAP = 48.0
RAND_RANGE = 1000


def entropy_weight_decay_func(epoch):
    # linear decay
    # return np.maximum(-0.05/(10**4) * epoch + 0.5, 0.1)
    return 0.5


def learning_rate_decay_func(epoch):
    # if epoch < 20000:
    #     rate = 0.0001
    # else:
    #     rate = 0.0001
    rate = 0.0001

    return rate


class Pensieve(BaseAbr):

    abr_name = "pensieve"

    def __init__(self, model_path: str = "", s_info: int = 6, s_len: int = 8,
                 a_dim: int = 6, plot_flag: bool = False, train_mode=False):
        """Penseive
        Input state matrix shape: [s_info, s_len]

        Args
            model_path: pretrained model_path.
            s_info: number of features in input state matrix.
            s_len: number of past chunks.
            a_dim: number of actions in action space.
        """
        self.s_info = s_info
        self.s_len = s_len
        self.a_dim = a_dim
        self.model_path = model_path
        self.jump_action = False
        if self.s_info == 6 and self.s_len == 8 and self.a_dim == 6:
            print('use original pensieve')
        elif self.s_info == 6 and self.s_len == 6 and self.a_dim == 3:
            self.jump_action = True
            print('use jump action')
        else:
            raise NotImplementedError
        self.plot_flag = plot_flag

        self.train_mode = train_mode
        if not self.train_mode:
            self.sess = tf.compat.v1.Session()
            self.actor = a3c.ActorNetwork(
                    self.sess,
                    state_dim=[self.s_info, self.s_len],
                    action_dim=self.a_dim,
                    bitrate_dim=BITRATE_DIM,
                )
            self.sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(max_to_keep=None)
            if self.model_path:
                saver.restore(self.sess, self.model_path)

    # def load_model(self, model_path: str):
    #     self.saver.restore(self.sess, model_path)
    #     print("Testing model restored.")

    def get_next_bitrate(self, state, last_bit_rate) -> int:
        bitrate, _ = self._get_next_bitrate(state, last_bit_rate, self.actor)
        return bitrate

    def _get_next_bitrate(self, state, last_bit_rate, actor):
        action_prob = actor.predict(state)
        action_cumsum = np.cumsum(action_prob)
        if self.jump_action:
            selection = (action_cumsum > np.random.randint(
                 1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            bit_rate = calculate_from_selection(selection, last_bit_rate)
        else:
            bit_rate = (
                action_cumsum
                > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
            ).argmax()
        return bit_rate, action_prob

    def _test(self, actor: a3c.ActorNetwork, trace: AbrTrace,
              video_size_file_dir: str, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if trace.name:
            log_name = os.path.join(save_dir, "{}_{}.csv".format(self.abr_name, trace.name))
        else:
            log_name = os.path.join(save_dir, "{}_log.csv".format(self.abr_name))
        abr_log = open(log_name, 'w')
        log_writer = csv.writer(abr_log, lineterminator='\n')
        log_writer.writerow(["timestamp", "bitrate", "buffer_size",
                             "rebuffering", "video_chunk_size", "delay",
                             "reward"])
        test_scheduler = TestScheduler(trace)
        net_env = Environment(test_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                              video_size_file_dir=video_size_file_dir)
        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(self.a_dim)
        if self.jump_action:
            selection = 0
            action_vec[selection] = 1
        else:
            action_vec[bit_rate] = 1

        s_batch = [np.zeros((self.s_info, self.s_len))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        final_reward = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            (
                delay,
                sleep_time,
                buffer_size,
                rebuf,
                video_chunk_size,
                next_video_chunk_sizes,
                end_of_video,
                video_chunk_remain,
            ) = net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = linear_reward(VIDEO_BIT_RATE[bit_rate], 
                                   VIDEO_BIT_RATE[last_bit_rate], rebuf)
            r_batch.append(reward)

            last_bit_rate = bit_rate

            log_writer.writerow([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate],
                                 buffer_size, rebuf, video_chunk_size, delay,
                                 reward])

            # log time_stamp, bit_rate, buffer_size, reward
            state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(
                np.max(VIDEO_BIT_RATE)
            )  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = (
                float(video_chunk_size) / float(delay) / M_IN_K
            )  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, : BITRATE_DIM] = (
                np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
            )  # mega byte
            state[5, -1] = np.minimum(
                video_chunk_remain, TOTAL_VIDEO_CHUNK,
            ) / float(TOTAL_VIDEO_CHUNK)
            
            bit_rate, action_prob = self._get_next_bitrate(
                np.reshape(state, (1, self.s_info, self.s_len)), 
                last_bit_rate, actor)

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                final_reward = sum(r_batch)
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(self.a_dim)
                if self.jump_action:
                    selection = 0
                    action_vec[selection] = 1
                else:
                    action_vec[bit_rate] = 1

                s_batch.append(np.zeros((self.s_info, self.s_len)))
                a_batch.append(action_vec)
                entropy_record = []

                break
        abr_log.close()
        if self.plot_flag:
            plot_abr_log(trace, log_name, save_dir)
        return final_reward

    def test(self, trace: AbrTrace, video_size_file_dir: str, save_dir: str):
        assert not self.train_mode
        return self._test(self.actor, trace, video_size_file_dir, save_dir)

    def test_on_traces(self, traces: List[AbrTrace], video_size_file_dir: str,
                       save_dirs: List[str]):
        assert not self.train_mode
        rewards = []
        for trace, save_dir in zip(traces, save_dirs):
            rewards.append(self.test(trace, video_size_file_dir, save_dir))
        return rewards

    def train(self, trace_scheduler, val_traces: List[AbrTrace],
              save_dir: str, num_agents: int, total_epoch: int,
              video_size_file_dir: str, model_save_interval: int = 1000,
              suffix: str = ""):
        assert self.train_mode

        # Visdom Settings
        # vis = visdom.Visdom()
        # assert vis.check_connection()
        plot_color = 'red'
        # Visdom Logs
        val_epochs = []
        val_mean_rewards = []
        average_rewards = []
        average_entropies = []

        logging.basicConfig(filename=os.path.join(save_dir, 'log_central'),
                            filemode='w', level=logging.INFO)

        # inter-process communication queues
        net_params_queues = []
        exp_queues = []
        for i in range(num_agents):
            net_params_queues.append(mp.Queue(1))
            exp_queues.append(mp.Queue(1))

        agents = []
        for i in range(num_agents):
            agents.append(mp.Process(
                target=agent,
                args=(TRAIN_SEQ_LEN, self.s_info, self.s_len, self.a_dim,
                      save_dir, i, net_params_queues[i], exp_queues[i], trace_scheduler,
                      video_size_file_dir, self.jump_action)))
        for i in range(num_agents):
            agents[i].start()
        with tf.Session() as sess, \
                open(os.path.join(save_dir, 'log_train'), 'w', 1) as log_central_file, \
                open(os.path.join(save_dir, 'log_val'), 'w', 1) as val_log_file:
            log_writer = csv.writer(log_central_file, delimiter='\t', lineterminator='\n')
            log_writer.writerow(['epoch', 'loss', 'avg_reward', 'avg_entropy'])
            val_log_writer = csv.writer(val_log_file, delimiter='\t', lineterminator='\n')
            val_log_writer.writerow(
                ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
                 'rewards_median', 'rewards_95per', 'rewards_max'])

            actor = a3c.ActorNetwork(sess,
                                     state_dim=[self.s_info, self.s_len],
                                     action_dim=self.a_dim,
                                     bitrate_dim=BITRATE_DIM)
                                     # learning_rate=args.ACTOR_LR_RATE)
            critic = a3c.CriticNetwork(sess,
                                       state_dim=[self.s_info, self.s_len],
                                       learning_rate=CRITIC_LR_RATE,
                                       bitrate_dim=BITRATE_DIM)

            logging.info('actor and critic initialized')
            # summary_ops, summary_vars = a3c.build_summaries()

            sess.run(tf.global_variables_initializer())
            # writer = tf.summary.FileWriter(save_dir, sess.graph)  # training monitor
            saver = tf.train.Saver(max_to_keep=None)  # save neural net parameters

            # restore neural net parameters
            if self.model_path:  # nn_model is the path to file
                saver.restore(sess, self.model_path)
                print("Model restored.")

            os.makedirs(os.path.join(save_dir, "model_saved"), exist_ok=True)

            epoch = 0

            # assemble experiences from agents, compute the gradients

            val_rewards = [self._test(
                actor, trace, video_size_file_dir=video_size_file_dir,
                save_dir=os.path.join(save_dir, "val_logs")) for trace in val_traces]
            val_mean_reward = np.mean(val_rewards)
            max_avg_reward = val_mean_reward

            val_log_writer.writerow(
                    [epoch, np.min(val_rewards),
                     np.percentile(val_rewards, 5), np.mean(val_rewards),
                     np.median(val_rewards), np.percentile(val_rewards, 95),
                     np.max(val_rewards)])
            val_epochs.append(epoch)
            val_mean_rewards.append(val_mean_reward)

            while epoch < total_epoch:
                start_t = time.time()
                # synchronize the network parameters of work agent
                actor_net_params = actor.get_network_params()
                critic_net_params = critic.get_network_params()
                for i in range(num_agents):
                    net_params_queues[i].put([actor_net_params, critic_net_params])

                # record average reward and td loss change
                # in the experiences from the agents
                total_batch_len = 0.0
                total_reward = 0.0
                total_td_loss = 0.0
                total_entropy = 0.0
                total_agents = 0.0

                # assemble experiences from the agents
                actor_gradient_batch = []
                critic_gradient_batch = []

                # linear entropy weight decay(paper sec4.4)
                entropy_weight = entropy_weight_decay_func(epoch)
                current_learning_rate = learning_rate_decay_func(epoch)

                for i in range(num_agents):
                    s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                    actor_gradient, critic_gradient, td_batch = \
                        a3c.compute_gradients(
                            s_batch=np.stack(s_batch, axis=0),
                            a_batch=np.vstack(a_batch),
                            r_batch=np.vstack(r_batch),
                            terminal=terminal, actor=actor,
                            critic=critic,
                            entropy_weight=entropy_weight)

                    actor_gradient_batch.append(actor_gradient)
                    critic_gradient_batch.append(critic_gradient)

                    total_reward += np.sum(r_batch)
                    total_td_loss += np.sum(td_batch)
                    total_batch_len += len(r_batch)
                    total_agents += 1.0
                    total_entropy += np.sum(info['entropy'])

                # compute aggregated gradient
                assert num_agents == len(actor_gradient_batch)
                assert len(actor_gradient_batch) == len(critic_gradient_batch)
                # assembled_actor_gradient = actor_gradient_batch[0]
                # assembled_critic_gradient = critic_gradient_batch[0]
                # for i in range(len(actor_gradient_batch) - 1):
                #     for j in range(len(assembled_actor_gradient)):
                #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
                #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
                # actor.apply_gradients(assembled_actor_gradient)
                # critic.apply_gradients(assembled_critic_gradient)
                for i in range(len(actor_gradient_batch)):
                    actor.apply_gradients(actor_gradient_batch[i], current_learning_rate)
                    critic.apply_gradients(critic_gradient_batch[i])

                # log training information
                epoch += 1
                avg_reward = total_reward / total_agents
                avg_td_loss = total_td_loss / total_batch_len
                avg_entropy = total_entropy / total_batch_len

                logging.info('Epoch: ' + str(epoch) +
                             ' TD_loss: ' + str(avg_td_loss) +
                             ' Avg_reward: ' + str(avg_reward) +
                             ' Avg_entropy: ' + str(avg_entropy))
                log_writer.writerow([epoch, avg_td_loss, avg_reward, avg_entropy])

                # summary_str = sess.run(summary_ops, feed_dict={
                #     summary_vars[0]: avg_td_loss,
                #     summary_vars[1]: avg_reward,
                #     summary_vars[2]: avg_entropy
                # })

                # writer.add_summary(summary_str, epoch)
                # writer.flush()

                if epoch % model_save_interval == 0:
                    # # Visdom log and plot

                    val_rewards = [self._test(
                        actor, trace, video_size_file_dir=video_size_file_dir,
                        save_dir=os.path.join(save_dir, "val_logs")) for trace in val_traces]
                    val_mean_reward = np.mean(val_rewards)

                    val_log_writer.writerow(
                            [epoch, np.min(val_rewards),
                             np.percentile(val_rewards, 5), np.mean(val_rewards),
                             np.median(val_rewards), np.percentile(val_rewards, 95),
                             np.max(val_rewards)])
                    val_epochs.append(epoch)
                    val_mean_rewards.append(val_mean_reward)
                    average_rewards.append(np.sum(avg_reward))
                    average_entropies.append(avg_entropy)

                    # suffix = args.start_time
                    # if args.description is not None:
                    #     suffix = args.description
                    # curve = dict(x=val_epochs, y=val_mean_rewards,
                    #              mode="markers+lines", type='custom',
                    #              marker={'color': plot_color,
                    #                      'symbol': 104, 'size': "5"},
                    #              text=["one", "two", "three"], name='1st Trace')
                    # layout = dict(title="Pensieve_Val_Reward " + suffix,
                    #               xaxis={'title': 'Epoch'},
                    #               yaxis={'title': 'Mean Reward'})
                    # vis._send(
                    #     {'data': [curve], 'layout': layout,
                    #      'win': 'Pensieve_val_mean_reward'})
                    # curve = dict(x=val_epochs, y=average_rewards,
                    #              mode="markers+lines", type='custom',
                    #              marker={'color': plot_color,
                    #                      'symbol': 104, 'size': "5"},
                    #              text=["one", "two", "three"], name='1st Trace')
                    # layout = dict(title="Pensieve_Training_Reward " + suffix,
                    #               xaxis={'title': 'Epoch'},
                    #               yaxis={'title': 'Mean Reward'})
                    # vis._send(
                    #     {'data': [curve], 'layout': layout,
                    #      'win': 'Pensieve_training_mean_reward'})
                    # curve = dict(x=val_epochs, y=average_entropies,
                    #              mode="markers+lines", type='custom',
                    #              marker={'color': plot_color,
                    #                      'symbol': 104, 'size': "5"},
                    #              text=["one", "two", "three"], name='1st Trace')
                    # layout = dict(title="Pensieve_Training_Mean Entropy " + suffix,
                    #               xaxis={'title': 'Epoch'},
                    #               yaxis={'title': 'Mean Entropy'})
                    # vis._send(
                    #     {'data': [curve], 'layout': layout,
                    #      'win': 'Pensieve_training_mean_entropy'})

                    # if val_mean_reward > max_avg_reward:
                    max_avg_reward = val_mean_reward
                    # Save the neural net parameters to disk.
                    save_path = saver.save(
                        sess,
                        os.path.join(save_dir, "model_saved", f"nn_model_ep_{epoch}.ckpt"))
                    logging.info("Model saved in file: " + save_path)

                end_t = time.time()
                # print(f'epoch{epoch-1}: {end_t - start_t}s')

        for tmp_agent in agents:
            tmp_agent.terminate()


def agent(train_seq_len: int, s_info: int, s_len: int, a_dim: int,
          save_dir: str, agent_id: int, net_params_queue: mp.Queue,
          exp_queue: mp.Queue, trace_scheduler, video_size_file_dir: str,
          jump_action: bool):
    """Agent method for A2C/A3C training framework.

    Args
        train_seq_len: train batch size
        net_params_queues: a queue for the transferring neural network
                           parameters between central agent and agent.
        exp_queues: a queue for the transferring experience/rollouts
                    between central agent and agent.
    """
    net_env = Environment(trace_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                          video_size_file_dir=video_size_file_dir,
                          random_seed=agent_id)
    with tf.compat.v1.Session() as sess:
        # , open(os.path.join(
        #     save_dir, f'log_agent_{agent_id}'), 'w') as log_file:

        # log_file.write('\t'.join(['time_stamp', 'bit_rate', 'buffer_size',
        #                'rebuffer', 'video_chunk_size', 'delay', 'reward',
        #                'epoch', 'trace_idx', 'mahimahi_ptr'])+'\n')
        actor = a3c.ActorNetwork(sess, state_dim=[s_info, s_len],
                                 action_dim=a_dim, bitrate_dim=BITRATE_DIM)
                                 # learning_rate=args.ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess, state_dim=[s_info, s_len],
                                   learning_rate=CRITIC_LR_RATE,
                                   bitrate_dim=BITRATE_DIM)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        selection = 0
        bit_rate = DEFAULT_QUALITY

        #action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
        action_vec = np.zeros(a_dim)
        if jump_action:
            action_vec[selection] = 1
        else:
            action_vec[bit_rate] = 1

        s_batch = [np.zeros((s_info, s_len))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        epoch = 0
        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = linear_reward(VIDEO_BIT_RATE[bit_rate], 
                                   VIDEO_BIT_RATE[last_bit_rate], rebuf)

            r_batch.append(reward)
            last_bit_rate = bit_rate

            # retrieve previous state
            state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be args.S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            # state[0 ,-1] = VIDEO_BIT_RATE[bit_rate] + \
            #          float(selection)  # last quality
            # state[0 ,-1] = VIDEO_BIT_RATE[bit_rate] / \
            #                VIDEO_BIT_RATE[last_bit_rate]  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :BITRATE_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, TOTAL_VIDEO_CHUNK) / float(TOTAL_VIDEO_CHUNK)

            # compute action probability vector
            action_prob = actor.predict(np.reshape(
                state, (1, s_info, s_len)))
            # bit_rate = action_prob.argmax()
            if np.isnan(action_prob[0, 0]) and agent_id == 0:
                print(epoch)
                print(state, "state")
                print(action_prob, "action prob")
                import pdb
                pdb.set_trace()
            action_cumsum = np.cumsum(action_prob)
            if jump_action:
                selection = (action_cumsum > np.random.randint(
                     1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                bit_rate = calculate_from_selection(selection, last_bit_rate)
            else:
                bit_rate = (
                    action_cumsum
                    > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
                ).argmax()

            # Note: we need to discretize the probability into 1/args.RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # report experience to the coordinator
            if len(r_batch) >= train_seq_len or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                #action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                action_vec = np.zeros(a_dim)
                if jump_action:
                    selection = 0
                    action_vec[selection] = 1
                else:
                    action_vec[bit_rate] = 1
                s_batch.append(np.zeros((s_info, s_len)))
                a_batch.append(action_vec)
                epoch += 1
                net_env.trace_scheduler.set_epoch(epoch)

            else:
                s_batch.append(state)

                #print(bit_rate)
                #action_vec = np.zeros(args.A_DIM)
                #action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                action_vec = np.zeros(a_dim)
                if jump_action:
                    action_vec[selection] = 1
                else:
                    action_vec[bit_rate] = 1
                #print(action_vec)
                a_batch.append(action_vec)


def calculate_from_selection(selected, last_bit_rate):
    # naive step implementation
    # action=0, bitrate-1; action=1, bitrate stay; action=2, bitrate+1
    if selected == 1:
        bit_rate = last_bit_rate
    elif selected == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    bit_rate = max(0, bit_rate)
    bit_rate = min(5, bit_rate)

    return bit_rate
