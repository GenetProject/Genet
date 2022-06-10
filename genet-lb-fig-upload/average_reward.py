from collections import deque


class AveragePerStepReward(object):
    def __init__(self, size):
        self.reward_record = deque(maxlen=size)
        self.time_record = deque(maxlen=size)
        self.reward_sum = 0
        self.time_sum = 0

    def add(self, reward, time):
        if len(self.reward_record) >= self.reward_record.maxlen:
            stale_reward = self.reward_record.popleft()
            stale_time = self.time_record.popleft()
            self.reward_sum -= stale_reward
            self.time_sum -= stale_time

        self.reward_record.append(reward)
        self.time_record.append(time)
        self.reward_sum += reward
        self.time_sum += time

    def add_list(self, list_reward, list_time):
        assert len(list_reward) == len(list_time)
        for i in range(len(list_reward)):
            self.add(list_reward[i], list_time[i])

    def add_list_filter_zero(self, list_reward, list_time):
        assert len(list_reward) == len(list_time)
        for i in range(len(list_reward)):
            if list_time[i] != 0:
                self.add(list_reward[i], list_time[i])
            else:
                assert list_reward[i] == 0

    def get_avg_per_step_reward(self):
        return float(self.reward_sum) / float(self.time_sum)
