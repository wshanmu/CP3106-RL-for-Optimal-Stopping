import pickle
import matplotlib.pyplot as plt
import numpy as np
import gym
import math
from scipy import signal


def normalization(data):
    _range = np.max(data) - np.min(data)
    normalized_data = (data - np.min(data)) / _range
    return normalized_data


def sigmoid(inx):
    if inx >= 0:
        return 1 / (1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))


class CryptoEnv(gym.Env):
    """
    An OpenAI Gym Style RL Environment for Optimal Stopping Problem with DCA 
    """
    def __init__(self, data, wnd_t=15, cycle_T=30):
        self.data = data
        self.wnd_t = wnd_t  # looking previous price data
        self.cycle_T = cycle_T  # Investment cycle
        self.action_space = gym.spaces.Discrete(2)  # buy & hold
        self.observation_space = gym.spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(1, self.wnd_t+2), dtype=np.float32)
        self.max_num_observations, self.original_wnd_lists, self.wnd_lists = self._prepare_data(self.data)
        self.num_episodes = self.get_total_num_episodes_per_epoch()
        self.begin_time = 0  # current episode begins at
        arr = np.arange(self.num_episodes)
        np.random.shuffle(arr)
        self.randomIndexes = arr.tolist()  # index for episode
        self.random_index = 0
        self.reset()
        self.original_price_list, self.normalized_price_list, self.refer_price = self.getPriceList()
        self.buy_time = cycle_T - 1

    def _prepare_data(self, data):
        """
        :param data: price list
        :return: no. of wnd sequences, all wnd sequences, and all normalized wnd sequences
        """
        original_wnd_lists = []
        if self.wnd_t > len(data):
            raise Exception('data must be longer than wnd_t. The length of data is: {}'.format(len(data)))
        max_num_observations = len(data) - self.wnd_t
        if self.cycle_T > max_num_observations:
            raise Exception('cycle_T must be longer than max number of obs. max_num_observations is: {}'.format(
                self.max_num_observations))
        original_wnd_lists.append([data[i:i + self.wnd_t] for i in range(max_num_observations)])
        original_wnd_lists = original_wnd_lists[0]
        wnd_lists = [normalization(i) for i in original_wnd_lists]

        return max_num_observations, original_wnd_lists, wnd_lists

    def get_total_num_episodes_per_epoch(self):
        total = 0
        total += self.max_num_observations - self.cycle_T
        return total

    def prepare_episodes(self):
        episodes = []
        for begin_time in range(self.max_num_observations - self.cycle_T):
            episode = self.wnd_lists[begin_time:begin_time + self.cycle_T]
            episodes.append(episode)
        return episodes

    def prepare_original_episodes(self):
        episodes = []
        for begin_time in range(self.max_num_observations - self.cycle_T):
            episode = self.original_wnd_lists[begin_time:begin_time + self.cycle_T]
            episodes.append(episode)
        return episodes

    def getPriceList(self):
        original_observations = self.original_wnd_lists[self.begin_time:self.begin_time + self.cycle_T]
        original_price_list = []
        for observation in original_observations:
            original_price_list.append(observation[-1])
        normalized_price_list = normalization(original_price_list)
        refer_value = original_observations[0][-2]
        return original_price_list, normalized_price_list, refer_value

    def reset(self):
        self.t = 0  # the data in current investment cycle
        self.buy_time = self.cycle_T - 1  # buy on which day
        self.done = False
        self.begin_time = self.randomIndexes[self.random_index]  # locate the episode
        self.random_index = (self.random_index + 1) % self.num_episodes
        self.observations = self.wnd_lists[self.begin_time:self.begin_time + self.cycle_T]
        self.original_price_list, self.normalized_price_list, self.refer_price = self.getPriceList()

        self.position_value = sigmoid(self.original_price_list[self.t] - self.refer_price)  # ref value

        self.remaining_time = (self.cycle_T - self.t)/self.cycle_T
        obs = np.concatenate(([self.position_value, self.remaining_time], self.observations[0]))
        return obs

    def step(self, action):
        reward = 0
        current_price = self.normalized_price_list[self.t]
        sorted_list = sorted(self.normalized_price_list)
        if (action == 1 or self.t == (self.cycle_T - 1)) and (self.done == False):
            self.sell_time = self.t
            if current_price == 0:
                current_price += 0.001
            if current_price == 1:
                current_price -= 0.001
            reward = math.log((1-current_price)/current_price)
            self.done = True
        elif action == 0:
            if current_price == 0:
                current_price += 0.001
            if current_price == 1:
                current_price -= 0.001
            reward = - 0.5 * math.log((1-current_price)/current_price)

        # set next time
        if self.t < (self.cycle_T - 1):
            self.t += 1

        self.position_value = sigmoid(self.original_price_list[self.t] - self.refer_price)
        self.remaining_time = (self.cycle_T - self.t) / self.cycle_T
        if self.done:
            obs = np.concatenate(([0, 0], [0] * self.wnd_t))
        else:
            obs = np.concatenate(([self.position_value, self.remaining_time], self.observations[self.t]))
        # info = {'time':self.sell_time, 'index': sorted_list.index(current_price)+1}
        info = {}
        return obs, reward, self.done, info  # obs, reward, done

    def seed(self, seed):
        np.random.seed(seed)

    def render(self):
        pass

    def close(self):
        pass