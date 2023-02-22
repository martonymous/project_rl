from TestEnv import HydroElectric_Test
from q_agent import *
import argparse
import matplotlib.pyplot as plt
import numpy.random as nprand

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='validate.xlsx') # Path to the excel file with the test data
parser.add_argument('--baseline', type=str, default='model') # Path to the excel file with the test data
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.excel_file)
total_reward = []
cumulative_reward = []

""" 
    OBSERVATIONS:
        water level
        price
        hour
        day of week
        day of year
        month
        year
"""
observation = env.observation()

with open('q_agent.pickle', 'rb') as f:
    agent = pickle.load(f)

agent.Actions = {
    0: -1,
    1: 0,
    2: 1,
}

water_level, electricity_cost, time_hour, time_day, day_of_year, time_month, year = [], [], [], [], [], [], []

def obs_to_df(obs):
    water_level.append(obs[0])
    electricity_cost.append(obs[1])
    time_hour.append(obs[2])
    time_day.append(obs[3])
    day_of_year.append(obs[4])
    time_month.append(obs[5])
    year.append(obs[6])

    if len(water_level) > 29:
        water_level.pop(0)
        electricity_cost.pop(0)
        time_hour.pop(0)
        time_day.pop(0)
        day_of_year.pop(0)
        time_month.pop(0)
        year.pop(0)

    df = pd.DataFrame(data={"water_level": water_level, "electricity_cost": electricity_cost, "time_hour": time_hour, "time_day": time_day, "day_of_year": day_of_year, "time_month": time_month, "year": year})
    return df

def get_features(df, n):
    df['rsi'] = 100 - (100 / (1 + df['electricity_cost'].diff(1).mask(df['electricity_cost'].diff(1) < 0, 0).ewm(alpha=1/n, adjust=False).mean() / df['electricity_cost'].diff(1).mask(df['electricity_cost'].diff(1) > 0, -0.0).abs().ewm(alpha=1/n, adjust=False).mean()))
    df['roc'] = (df['electricity_cost'].shift(n) - df['electricity_cost']) / df['electricity_cost'].shift(n) * 100
    df['weekend'] = df['time_day'].apply(lambda x: 1 if x > 4 else 0)
    return df

def get_state(df):
    state = {}
    state["time_hour"] = int(df["time_hour"].iloc[-1]-1)
    state["water_level"] = np.digitize(df["water_level"].iloc[-1], agent.bin_water_level)-1
    state["electricity_cost"] = np.digitize(df["electricity_cost"].iloc[-1], agent.bin_prices)-1
    state["time_day"] = int(df["time_day"].iloc[-1]-1)
    state["time_weekend"] = int(df["weekend"].iloc[-1])
    state["time_month"] = int(df["time_month"].iloc[-1]-1)
    return state

for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
    # Choose a random action between -1 (full capacity sell) and 1 (full capacity pump)
    # action = env.continuous_action_space.sample()
    # Or choose an action based on the observation using your RL agent!:
    # action = RL_agent.act(observation)
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    
    agent.observations = obs_to_df(obs=observation)
    agent.observations = get_features(df=agent.observations, n=6)
    state = get_state(agent.observations)
    
    if args.baseline == 'baseline':
        randnumber = nprand.uniform(0,1)

        if state["time_hour"] in [9,10,11,12,18,19] and randnumber > 0:
            action = -1
        elif state["time_hour"] in [1,2,3,4,5,6] and randnumber > 0:
            action = 1
        else:
            action = 0
    else:
        a = agent.Qtable[
            state["time_hour"],
            state["water_level"],
            state["electricity_cost"],
            state["time_weekend"],
            state["time_month"], :
        ]
        action = agent.Actions[np.argmax(a)]

    next_observation, reward, terminated, truncated, info = env.step(action)
    if next_observation[5] != observation[5]:
        print(f"\nEvaluating! It is now Year {next_observation[6]} and Month {next_observation[5]}\nCumulative reward so far: {cumulative_reward[-1]}")
    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))

    done = terminated or truncated
    observation = next_observation

    if done:
        print('Total reward: ', sum(total_reward))
        # Plot the cumulative reward over time
        plt.plot(cumulative_reward)
        plt.xlabel('Time (Hours)')
        plt.show()




