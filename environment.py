import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import pandas as pd
import random

class DamWorldEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, observation_data: pd.DataFrame):
        self.data = observation_data
        self.Actions = {
            "sell_1": -18000,
            "sell_2": -10000,
            "sell_3": -4000,
            "sell_4": -2000,
            "hold": 0,
            "buy_4": 2000,
            "buy_3": 4000,
            "buy_2": 10000,
            "buy_1": 18000,
        }

        # self.flow_multiplier = {
        #     0: 1,
        #     1: 2,
        #     2: 4,
        #     3: 9,
        #     4: 18
        # }
        # self.flow_rate = 1000

        self.sell_efficiency = 0.9
        self.buy_efficiency = 0.8

        self.time_hour_dim = 24
        self.time_day_dim = 7
        self.time_week_dim = 53
        self.time_month_dim = 12
        self.water_level_dim = 21
        self.water_capacity = 100000
        # for converting water level (m³) to MWh
        #  pot. energy of 1 m³ = mass *    g *  h * (Joule to MWH factor)
        self.conversion_factor = 1000 * 9.81 * 30 * (2 + (7/9)) * (10 ** -10)

        self.action_space = spaces.Discrete(len(self.Actions), 42)
        self.observation_space = spaces.Dict(
            {
                "time_hour": spaces.Discrete(self.time_hour_dim, 42, 0),
                "time_day": spaces.Discrete(self.time_day_dim, 42, 0),
                "time_week": spaces.Discrete(self.time_week_dim, 42, 0),
                "time_month": spaces.Discrete(self.time_month_dim, 42, 0),

                "water_level": spaces.Discrete(self.water_level_dim, 42, 0),
                "electricity_cost": spaces.Box(0, self.data["prices"].max())
            }
        )

    def _get_obs(self):
        self.hour = self.data["hour"].iloc[self.index]
        self.day = self.data["day"].iloc[self.index]
        self.week = self.data["week"].iloc[self.index]
        self.month = self.data["month"].iloc[self.index]
        self.electricity_cost = self.data["prices"].iloc[self.index]

        return {
            "time_hour": self.hour, 
            "time_day": self.day,
            "time_week": self.week,
            "time_month": self.month,
            "water_level": self.water_level,
            "electricity_cost": self.electricity_cost,
            "cash": self.cash,
            "value": self.value
        }
    
    def _get_info(self):
        return {
            "profit": (self.cash - self.starting_cash),
            "unrealized_profit": self.water_level * self.electricity_cost * self.sell_efficiency * self.conversion_factor,
            "total_value": (self.cash - self.starting_cash) + (self.water_level * self.electricity_cost * self.sell_efficiency * self.conversion_factor)
        }
    
    def step(self, action, terminated=False):
        info = self._get_info()

        previous_total_value = info["total_value"]
        previous_cash = self.cash

        # first check if simulation terminates, otherwise move index and perform action
        if (self.index+1) == self.data.shape[0] or (self.water_level == 0 and self.cash == 0):
            terminated = True
        else:
            # otherwise continue
            self.index += 1

            # flow_mult = self.flow_multiplier[action[1]]

            # we can only sell if there is water in the dam
            if (action < 0) and (self.water_level != 0):
                if self.water_level > abs(action):
                    self.cash += self.electricity_cost * self.sell_efficiency * abs(action) * self.conversion_factor
                    self.water_level += action
                else:
                    self.cash += self.electricity_cost * self.sell_efficiency * self.water_level * self.conversion_factor
                    self.water_level = 0
                    
            # we can only buy if we have cash and if dam is not full
            elif (action > 0) and (self.water_level < self.water_capacity) and (self.cash > (self.electricity_cost * self.sell_efficiency * (self.water_capacity - self.water_level) * self.conversion_factor)):
                if (self.water_capacity - self.water_level) > abs(action):
                    self.cash -= self.electricity_cost * self.buy_efficiency * abs(action) * self.conversion_factor
                    self.water_level += action
                else:
                    self.cash -= self.electricity_cost * self.sell_efficiency * (self.water_capacity - self.water_level) * self.conversion_factor
                    self.water_level = self.water_capacity
                    
        observation = self._get_obs()
        info = self._get_info()
        self.value = info["total_value"]
        
        reward = self.value - previous_total_value

        # if self.value > previous_total_value:
        #     reward = 1
        # else:
        #     reward = -1

        return observation, reward, terminated, False, info

    def reset(self, val=False):
        if val:
            self.index = 0
        else:
            self.index = random.randint(0, len(self.data) - 50)

        self.hour = self.data["hour"].iloc[self.index]
        self.day = self.data["day"].iloc[self.index]
        self.week = self.data["week"].iloc[self.index]
        self.month = self.data["month"].iloc[self.index]

        self.water_level = 50000  # half of maximum water level
        self.starting_cash = 0  # (arbitrary) amount of cash
        self.cash = self.starting_cash
        self.electricity_cost = self.data["prices"].iloc[self.index]
        self.value = self.starting_cash + (self.electricity_cost * self.water_level * self.conversion_factor)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

if __name__ == "__main__":

    data = pd.read_csv('data/train_processed.csv')
    env = DamWorldEnv(observation_data=data)

    # test environment
    obs, inf = env.reset()
    print(obs)
    for _ in range(10):
        obs, reward, term, trunc, inf = env.step(env.action_space.sample())
    print(obs)
    print(reward)