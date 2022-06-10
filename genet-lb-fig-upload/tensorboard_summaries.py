import tensorflow as tf


def build_tensorboard_summaries(var_list):
    summary_vars = []

    for var in var_list:
        tf_var = tf.Variable(0.)
        tf.summary.scalar(var, tf_var)
        summary_vars.append(tf_var)

    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def build_load_balance_tf_summaries():
    action_loss = tf.Variable(0.)
    tf.summary.scalar("Action loss", action_loss)
    entropy = tf.Variable(0.)
    tf.summary.scalar("Entropy", entropy)
    value_loss = tf.Variable(0.)
    tf.summary.scalar("Value loss", value_loss)
    eps_length = tf.Variable(0.)
    tf.summary.scalar("Episode length", eps_length)
    average_reward = tf.Variable(0.)
    tf.summary.scalar("Average number of concurrent jobs", average_reward)
    sum_rewards = tf.Variable(0.)
    tf.summary.scalar("Sum of rewards", sum_rewards)
    eps_duration = tf.Variable(0.)
    tf.summary.scalar("Episode duration in seconds", eps_duration)
    entropy_weight = tf.Variable(0.)
    tf.summary.scalar("Entropy weight", entropy_weight)
    reset_prob = tf.Variable(0.)
    tf.summary.scalar("Reset probability", reset_prob)
    num_stream_jobs = tf.Variable(0.)
    tf.summary.scalar("Number of stream jobs", num_stream_jobs)
    reset_hit = tf.Variable(0.)
    tf.summary.scalar("Fraction of reset hit", reset_hit)
    eps_finished_jobs = tf.Variable(0.)
    tf.summary.scalar("Number of finished jobs in an episode", eps_finished_jobs)
    eps_unfinished_jobs = tf.Variable(0.)
    tf.summary.scalar("Number of unfinished jobs", eps_unfinished_jobs)
    eps_finished_work = tf.Variable(0.)
    tf.summary.scalar("Number of finished total work in an episode", eps_finished_work)
    eps_unfinished_work = tf.Variable(0.)
    tf.summary.scalar("Number of unfinished work", eps_unfinished_work)
    average_job_duration = tf.Variable(0.)
    tf.summary.scalar("Average job duration", average_job_duration)

    summary_vars = [action_loss, entropy, value_loss, eps_length, \
                    average_reward, sum_rewards, eps_duration, \
                    entropy_weight, reset_prob, num_stream_jobs, \
                    reset_hit, eps_finished_jobs, eps_unfinished_jobs, \
                    eps_finished_work, eps_unfinished_work, average_job_duration]

    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
