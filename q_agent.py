from environment import DamWorldEnv
import pandas as pd
import numpy as np
import matplotlib as plt
import gymnasium as gym


class QAgent():
    
    def __init__(self, env: DamWorldEnv, data: pd.DataFrame, bin_size = 101, discount_rate = 0.95):
        
        '''
        Params:
        
        env_name = name of the specific environment that the agent wants to solve
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
        
        # create an environment
        self.env = DamWorldEnv(observation_data=data)
        
        # Set the discount rate and bin size
        self.discount_rate = discount_rate
        self.bin_size = bin_size
        
        # The algorithm has  3 different actions, as well as varying flow rates (to move more or less water at a time)
        #0: sell water-storage for electricity
        #1: buy water-storage
        #2: hold on
        self.action_space1 = self.env.action_space.nvec[0]
        self.action_space2 = self.env.action_space.nvec[1]
    
        # Create bins for continuous observation features, i.e. price and water level
        self.bin_prices = np.linspace(0, self.env.data["prices"].max(), self.bin_size)
        self.bin_water_level = np.linspace(0, 100000, self.env.water_level_dim+1)
        self.bins = [self.bin_prices, self.bin_water_level]
    
    def discretize_state(self, state):
        
        '''
        Params:
        state = state observation that needs to be discretized
        
        Returns:
        discretized state
        '''
        self.state = state
        binned_states = [
            "electricity_cost",
            "water_level",
        ]
    
        for i in range(len(self.bins)):
            state[binned_states[i]] = np.digitize(self.state[binned_states[i]], self.bins[i])-1

        return state
    
    def create_Q_table(self):
        self.state_space = self.bin_size - 1
        #Initialize all values in the Q-table to zero
        self.Qtable = np.zeros((
            self.bin_size, 
            self.env.water_level_dim, 
            self.env.time_hour_dim, 
            self.env.time_day_dim, 
            self.env.time_week_dim, 
            self.env.time_month_dim, 
            self.action_space1, 
            self.action_space2
        ))
        

    def train(self, simulations, learning_rate, episodes = 1000, epsilon = 0.05, epsilon_decay = 1000, adaptive_epsilon = False, 
              adapting_learning_rate = False):
        
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
        
        self.rewards = []
        self.average_rewards = []
        
        #Call the Q table function to create an initialized Q table
        self.create_Q_table()
        
        #Set epsilon rate, epsilon decay and learning rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        #Set start epsilon, so here we want a starting exploration rate of 1
        self.epsilon_start = 1
        self.epsilon_end = 0.05
        
        #If we choose adaptive learning rate, we start with a value of 1 and decay it over time!
        if adapting_learning_rate:
            self.learning_rate = 1
        
        for i in range(simulations):
            
            if i % 250 == 0:
                print(f'Please wait, the algorithm is learning! The current simulation is {i}')
            #Initialize the state
            state = self.env.reset()[0]   # reset returns a dict, need to take the 0th entry.
        
            #Set a variable that flags if an episode has terminated
            done = False
        
            #Discretize the state space
            
            state = self.discretize_state(state)
            
            #Set the rewards to 0
            total_rewards = 0
            
            #If adaptive epsilon rate
            if adaptive_epsilon:
                self.epsilon = np.interp(i, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
                
                #Logging just to check it decays as we want it to do, we just print out the first three statements
                if i % 500 == 0 and i <= 1500:
                    print(f"The current epsilon rate is {self.epsilon}")
                
            #Loop until an episode has terminated
            episode = 0
            while (not done) or (episode < episodes):
                episode += 1
                
                # Epsilon greedy
                #Pick random action
                if np.random.uniform(0,1) > 1-self.epsilon:
                    #This picks a random action from 0,1,2
                    action = self.env.action_space.sample()

                #Pick a greedy action
                else:
                    a = self.Qtable[
                        state["electricity_cost"], 
                        state["water_level"], 
                        state["time_hour"], 
                        state["time_day"], 
                        state["time_week"], 
                        state["time_month"], :, :
                        ]
                    action = np.unravel_index(np.argmax(a), a.shape)
                    
                #Now sample the next_state, reward, done and info from the environment
                
                next_state, reward, terminated, truncated, info = self.env.step(action) # step returns 5 outputs
                done =  terminated or truncated
                
                #Now discretize the next_state
                next_state = self.discretize_state(next_state)
                
                #Target value 
                Q_target = (
                    reward + self.discount_rate*np.max(self.Qtable[
                        state["electricity_cost"], 
                        state["water_level"], 
                        state["time_hour"], 
                        state["time_day"], 
                        state["time_week"], 
                        state["time_month"]
                    ])
                )
                
                #Calculate the Temporal difference error (delta)
                delta = self.learning_rate * (Q_target - self.Qtable[
                    state["electricity_cost"], 
                    state["water_level"], 
                    state["time_hour"], 
                    state["time_day"], 
                    state["time_week"], 
                    state["time_month"],
                    action[0],
                    action[1]
                ])
                
                #Update the Q-value
                self.Qtable[
                    state["electricity_cost"], 
                    state["water_level"], 
                    state["time_hour"], 
                    state["time_day"], 
                    state["time_week"], 
                    state["time_month"],
                    action[0],
                    action[1]
                ] = self.Qtable[
                    state["electricity_cost"], 
                    state["water_level"], 
                    state["time_hour"], 
                    state["time_day"], 
                    state["time_week"], 
                    state["time_month"],
                    action[0],
                    action[1]
                ] + delta
                
                #Update the reward and the hyperparameters
                total_rewards += reward
                state = next_state
                
            
            if adapting_learning_rate:
                self.learning_rate = self.learning_rate/np.sqrt(i+1)
            
            self.rewards.append(total_rewards)
            
            #Calculate the average score over 100 episodes
            if i % 100 == 0:
                self.average_rewards.append(np.mean(self.rewards))
                
                #Initialize a new reward list, as otherwise the average values would reflect all rewards!
                self.rewards = []
        
        print('The simulation is done!')
        
    def visualize_rewards(self):
        plt.figure(figsize =(7.5,7.5))
        plt.plot(100*(np.arange(len(self.average_rewards))+1), self.average_rewards)
        plt.axhline(y = -110, color = 'r', linestyle = '-')
        plt.title('Average reward over the past 100 simulations', fontsize = 10)
        plt.legend(['Q-learning performance','Benchmark'])
        plt.xlabel('Number of simulations', fontsize = 10)
        plt.ylabel('Average reward', fontsize = 10)


if __name__ == "__main__":
    
    data = pd.read_csv('data/train_processed.csv')
    agent_standard_greedy = QAgent(DamWorldEnv, data=data)
    agent_standard_greedy.train(2000, 0.1)