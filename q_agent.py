from environment import DamWorldEnv
from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from multiprocessing import Pool, cpu_count
import pickle, os


class QAgent():
    
    def __init__(self, data: pd.DataFrame, val_data: pd.DataFrame, bin_size = 26, discount_rate = 0.9):

        # create a train and validation environment
        self.env = DamWorldEnv(observation_data=data)
        self.val_env = DamWorldEnv(observation_data=val_data)
        
        # Set the discount rate and bin size
        self.discount_rate = discount_rate
        self.bin_size = bin_size
        
        # The algorithm has  3 different actions, as well as varying flow rates (to move more or less water at a time)
        #0: sell water-storage for electricity
        #1: buy water-storage
        #2: hold on
        self.action_space = self.env.action_space.n
    
        # Create bins for continuous observation features
        self.bin_prices = np.geomspace(1, self.env.data["prices"].max(), self.bin_size-1)
        self.bin_prices = np.insert(self.bin_prices, 0, 0)
        self.bin_water_level = np.linspace(0, 100000, self.env.water_level_dim)
        self.bin_rsi = np.linspace(self.env.data["prices"].min(), self.env.data["prices"].max(), self.env.rsi_dim)
        self.bin_roc = np.linspace(self.env.data["weekend"].min(), self.env.data["weekend"].max(), self.env.roc_dim)
        self.bins = [self.bin_prices, self.bin_water_level]
    
    def discretize_state(self, state):
        
        self.state = state
        binned_states = [
            "electricity_cost",
            "water_level",
        ]
    
        for i in range(len(self.bins)):
            state[binned_states[i]] = np.digitize(self.state[binned_states[i]], self.bins[i])-1

        return state
    
    def create_Q_table(self, init_val):
        #Initialize all values in the Q-table 
        self.Qtable = np.zeros((
            self.env.time_hour_dim,
            self.env.water_level_dim,
            self.bin_size,
            self.env.time_weekend_dim,
            self.env.time_month_dim,
            self.action_space
        ))
        self.Qtable = self.Qtable + init_val
    
    def visualize_rewards(self):
        plt.figure(figsize =(7.5,7.5))
        plt.plot(20*(np.arange(len(self.average_rewards))+1), self.average_rewards)
        plt.title('Average reward over the past 100 simulations', fontsize = 10)
        plt.legend(['Q-learning performance','Benchmark'])
        plt.xlabel('Number of simulations', fontsize = 10)
        plt.ylabel('Average reward', fontsize = 10)
        plt.show()

def simulate(agent, i, episodes = 1000):
    # if i % 25 == 0:
    #     print(f'Please wait, the algorithm is learning! The current simulation is {i}')
    if not agent.alr_start:
        agent.alr_start = 0
    if agent.adapting_learning_rate and i > agent.alr_start:
        learning_rate = agent.alr_val/np.sqrt(((i-agent.alr_start)/agent.alr_factor)+1)
    else:
        learning_rate = agent.learning_rate

    # If adaptive epsilon rate
    if agent.adaptive_epsilon:
        agent.epsilon = np.interp(i, [agent.epsilon_decay_start, agent.epsilon_decay_end], [agent.epsilon_start, agent.epsilon_end])

    # If adaptive epsilon rate
    if agent.adaptive_discount:
        agent.discount_rate = np.interp(i, [agent.discount_start, agent.discount_end], [agent.discount_start_value, agent.discount_end_value])

    if agent.phase_bins:
        phase_nr = np.digitize(i, agent.phase_bins)-1
    else:
        phase_nr = -1

    # Initialize the state
    state = agent.env.reset()[0]   # reset returns a dict, need to take the 0th entry.

    # Set a variable that flags if an episode has terminated
    done = False

    # Discretize the state space
    state = agent.discretize_state(state)
    
    # Set the rewards to 0
    total_rewards = 0
        
    # Loop until an episode has terminated
    episode = 0
    while not done:        
        episode += 1
        
        # Epsilon greedy
        # Pick random action
        if np.random.uniform(0,1) > 1-agent.epsilon:
            # This picks a random action from 0,1,2
            action = agent.env.action_space.sample()

        # Pick a greedy action
        else:
            a = agent.Qtable[state["time_hour"], state["water_level"], state["electricity_cost"], state["time_weekend"], state["time_month"], :]
            action = np.argmax(a)
            
        # Now sample the next_state, reward, done and info from the environment        
        next_state, reward, terminated, truncated, info = agent.env.step(action) # step returns 5 outputs

        done =  terminated or truncated
        if episode == episodes:
            done = True
        
        # Now discretize the next_state
        next_state = agent.discretize_state(next_state)
        
        # Target value
        a = agent.Qtable[next_state["time_hour"], next_state["water_level"], next_state["electricity_cost"], next_state["time_weekend"], next_state["time_month"]]
        Q_target = (reward + agent.discount_rate*np.max(a))
        
        # Calculate the Temporal difference error (delta)
        delta = learning_rate * (Q_target - agent.Qtable[state["time_hour"], state["water_level"], state["electricity_cost"], state["time_weekend"], state["time_month"], action])
        
        # Update the Q-value
        # if phase_nr == 0:
        #     agent.Qtable[state["water_level"], state["electricity_cost"], state["time_weekend"], :, :, :, action] =agent.Qtable[state["water_level"], state["electricity_cost"], state["time_weekend"], :, :, :, action] + delta
        if phase_nr == 0:
            agent.Qtable[state["time_hour"], state["water_level"], state["electricity_cost"], :, :, action] = agent.Qtable[state["time_hour"], state["water_level"], state["electricity_cost"], :, :, action] + delta
        if phase_nr == 1:
            agent.Qtable[state["time_hour"], state["water_level"], state["electricity_cost"], state["time_weekend"], :, action] = agent.Qtable[state["time_hour"], state["water_level"], state["electricity_cost"], state["time_weekend"], :, action] + delta
        else:
            agent.Qtable[state["time_hour"], state["water_level"], state["electricity_cost"], state["time_weekend"], state["time_month"], action] = agent.Qtable[state["time_hour"], state["water_level"], state["electricity_cost"], state["time_weekend"], state["time_month"], action] + delta

        # Update the reward and the hyperparameters
        total_rewards += reward
        state = next_state
    
    if i % 100 == 0:
        print(f"\nCurrent simulation:  {i}")
        cash = []
        agent.val_env.reset(val=True)

        for i in range(len(agent.val_env.data)):
            state = agent.val_env._get_obs()
            state = agent.discretize_state(state)
            
            a = agent.Qtable[
                state["time_hour"],
                state["water_level"],
                state["electricity_cost"],
                state["time_weekend"],
                state["time_month"], :
            ]
            action = np.argmax(a)

            next_state, reward, terminated, truncated, info = agent.val_env.step(action) # step returns 5 outputs
            cash.append(info["profit"])

        agent.validation_scores.append(cash[-1])

    #Calculate the average score over 100 episodes
    if i % 20 == 0:
        agent.average_rewards.append(np.mean(agent.rewards))
        
        #Initialize a new reward list, as otherwise the average values would reflect all rewards!
        agent.rewards = []

def train(agent, unique_id, simulations, learning_rate=0.1, episodes=1000, epsilon=0.5, discount_start=0, discount_end=100000, epsilon_decay_start = 100000, epsilon_decay_end = 200000, adaptive_discount=True, adaptive_epsilon = False, 
              adapting_learning_rate = False, alr_start=None, alr_factor=100, alr_val=0.5, phase_bins=[0, 50000, 100000, 150000, 200000, 250000], multiprocessing=False, max_workers=None, checkpoints=True, checkpoint_save_every = 2000):
        
        '''
        Params:
        simulations = number of episodes of a game to run
        learning_rate = learning rate for the update equation
        epsilon = epsilon value for epsilon-greedy algorithm
        epsilon_decay = number of full episodes (games) over which the epsilon value will decay to its final value
        adaptive_epsilon = boolean that indicates if the epsilon rate will decay over time or not
        adapting_learning_rate = boolean that indicates if the learning rate should be adaptive or not
        
        '''
        
        #Initialize variables that keep track of the rewards
        
        agent.rewards = []
        agent.average_rewards = []
        agent.validation_scores = []
        
        #Call the Q table function to create an initialized Q table
        agent.create_Q_table(init_val=100)
        
        #Set epsilon rate, epsilon decay, learning rate, and discount rate
        agent.adaptive_discount = adaptive_discount
        agent.adapting_learning_rate = adapting_learning_rate
        agent.adaptive_epsilon = adaptive_epsilon
        agent.alr_start = alr_start
        agent.alr_factor = alr_factor
        agent.alr_val = alr_val

        agent.learning_rate = learning_rate
        agent.epsilon = epsilon
        agent.epsilon_decay_start = epsilon_decay_start
        agent.epsilon_decay_end = epsilon_decay_end

        agent.discount_start = discount_start
        agent.discount_end = discount_end
        
        #Set start epsilon, so here we want a starting exploration rate of 1
        agent.epsilon_start = 1
        agent.epsilon_end = 0.2
        agent.discount_start_value = 0.9
        agent.discount_end_value = 0.999

        agent.phase_bins = phase_bins
        
        if multiprocessing:
            args = [(agent, x, episodes, 0.1) for x in range(simulations)]
            if max_workers == None:
                max_workers = cpu_count()-2
            print(max_workers)
            with Pool(processes=(max_workers)) as p:
                results = p.starmap(simulate, args)
        else:
            for i in range(simulations):
                simulate(agent, i, episodes)
                if checkpoints and (i % checkpoint_save_every == 0):
                    if phase_bins:
                        save_model(q_agent, np.digitize(i, phase_bins), unique_id)
                    else:
                        save_model(q_agent, 'FINAL', unique_id)
            if not checkpoints:
                save_model(q_agent, i, unique_id)

        print('The simulation is done!')

def evaluate(agent, val_data):
    # agent.visualize_rewards()
    rewards = 0
    action_sequence, cash, water_level, all_rewards = [], [], [], []

    agent.env = DamWorldEnv(observation_data=val_data)
    agent.env.reset(val=True)

    for i in range(len(val_data)):
        state = agent.env._get_obs()
        state = agent.discretize_state(state)
        
        a = agent.Qtable[
            state["time_hour"],
            state["water_level"],
            state["electricity_cost"],
            state["time_weekend"],
            state["time_month"], :
        ]
        action = np.argmax(a)
        action_sequence.append(action)

        next_state, reward, terminated, truncated, info = agent.env.step(action) # step returns 5 outputs
        rewards += reward
        water_level.append(next_state["water_level"])
        all_rewards.append(rewards)
        cash.append(info["profit"])

    # agent.visualize_rewards()

    print('The evaluation is done!')
    return action_sequence, cash, all_rewards, water_level
        
if __name__ == "__main__":
    eval = False

    if not eval:

        Seed = 13
        np.random.seed(Seed)
        UNIQUE_RUN_ID = f'{Seed}' # there' an option to make a unique id with a utility function in utils.py
        make_directory_for_run(UNIQUE_RUN_ID)

        data = pd.read_csv('train.csv')
        val_data = pd.read_csv('val.csv')
        q_agent = QAgent(data=data, val_data=val_data, discount_rate=0.99999)
        train(q_agent, UNIQUE_RUN_ID, 
            simulations=100001, 
            learning_rate=0.1, 
            episodes=2688,
            epsilon=1, 

            adaptive_discount=True, 
            adapting_learning_rate=True, 
            adaptive_epsilon=True,

            discount_start=0, 
            discount_end=100000,

            epsilon_decay_start=80000, 
            epsilon_decay_end=100000,

            alr_start=0, 
            alr_factor=200,
            alr_val=0.1,

            phase_bins=[0, 20001, 40001, 60001, 80001, 100001],

            max_workers=8, 
            multiprocessing=False, 
            checkpoint_save_every=10)

    else:

        list_of_files = glob.glob('./runs/*')
        list_of_files.sort(key=os.path.getctime)
        latest_folder = max(list_of_files, key=os.path.getctime)

        list_of_files = glob.glob(f'{latest_folder}/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)

        val_data = pd.read_csv('data/val.csv')

        with open(latest_file,'rb') as f:
            trained_agent = pickle.load(f)

        actions, cash, rewards, water_level = evaluate(trained_agent, val_data)

        df = pd.DataFrame({"prices": val_data["prices"], 'actions': actions, "cash": cash, "water_level": water_level, "rewards": rewards})
        df.to_csv('evaluated_model.csv')