import os
import sys
import itertools
import numpy as np
from collections import OrderedDict


def aggregate_gradients(gradients):

    ground_gradients = [np.zeros(g.shape) for g in gradients[0]]
    for gradient in gradients:
        for i in range(len(ground_gradients)):
            ground_gradients[i] += gradient[i]
    return ground_gradients


def compute_CDF(arr, num_bins=100):
    """
    usage: x, y = compute_CDF(arr):
           plt.plot(x, y)
    """
    values, base = np.histogram(arr, bins=num_bins)
    cumulative = np.cumsum(values)
    return base[:-1], cumulative / float(cumulative[-1])


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def decrease_var(var, min_var, decay_rate):
    if var - decay_rate >= min_var:
        var -= decay_rate
    else:
        var = min_var
    return var


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def generate_coin_flips(p, sim=False):
    # generate coin flip until first head, with Pr(head) = p
    # this follows a geometric distribution
    if p == 0:
        # infinite sequence
        return np.inf

    if sim:
        # actually simulate the coin flip
        flip_counts = 0
        while True:
            flip_counts += 1
            if np.random.rand() < p:
                break
    else:
        # use geometric distribution
        flip_counts = np.random.geometric(p)

    return flip_counts


def increase_var(var, max_var, increase_rate):
    if var + increase_rate <= max_var:
        var += increase_rate
    else:
        var = max_var
    return var


class OrderedSet(object):
    def __init__(self, contents=()):
        self.set = OrderedDict((c, None) for c in contents)

    def __contains__(self, item):
        return item in self.set

    def __iter__(self):
        return self.set.iterkeys()

    def __len__(self):
        return len(self.set)

    def add(self, item):
        self.set[item] = None

    def clear(self):
        self.set.clear()

    def pop(self):
        item = next(iter(self.set))
        del self.set[item]
        return item

    def remove(self, item):
        del self.set[item]

    def to_list(self):
        return [k for k in self.set]


def progress_bar(count, total, status='', pattern='|', back='-', bar_len=40):
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = pattern * filled_len + back * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s  %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

    if count == total:
        print('')


def split_list(lst, split_pts):
    split_idx = np.cumsum(split_pts)
    assert all(i <= len(lst) for i in split_pts)
    assert split_idx[0] > 0
    assert split_idx[-1] == len(lst)
    sub_lists = []
    sub_lists.append(lst[:split_idx[0]])
    for i in range(len(split_idx) - 1):
        sub_lists.append(
            lst[split_idx[i]:split_idx[i + 1]])
    return sub_lists


def squash(inputs, shift=0.0, scale=1.0):
    # inputs is np.array
    # squash to [-1, 1]
    x = (inputs - shift) / float(scale)
    return 2.0 / (1.0 + np.exp(-x)) - 1.0
