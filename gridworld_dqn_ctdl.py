
from collections import deque
import numpy as np
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #scoring
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from subprocess import run
import sys
#size of the grid
X_LEN = 10
Y_LEN = 10

#Location of the start state
X_START =  8 #random.randint(0, X_LEN - 1)
Y_START =  4 #random.randint(0, Y_LEN - 1)

#Location of the goal state
X_GOAL = 5 #random.randint(0, X_LEN - 1)
Y_GOAL = 9 #random.randint(0, Y_LEN - 1)

def print_out_everything(episode_rewards,time_step_episode, episodes_num, paths_taken ):
    #get the number of steps taken for every path taken and put into an array
    lengths_of_paths = np.zeros(len(paths_taken))
    for k in range(len(paths_taken)):
      lengths_of_paths[k] = len(paths_taken[k])

    #Print out the episode rewards in a graph
    plt.plot(episode_rewards)
    plt.ylabel('Episode Rewards')
    plt.xlabel('Episode')
    plt.show()

    #Print out a curve of the relation between episodes and time steps
    plt.plot( time_step_episode, episodes_num)
    plt.ylabel('Episode ')
    plt.xlabel('Time Step')
    plt.show()

    #Print out all the number of steps it takes to reach the optimal policy
    print("Number of steps in the smallest path:")
    print(lengths_of_paths[np.argmin(lengths_of_paths)])

    #print out a graph of the best path taken  - best policy
    plot_out = paths_taken[np.argmin(lengths_of_paths)]
    col = np.zeros(len(plot_out))
    row = np.zeros(len(plot_out))
    for i in range(len(plot_out)):
      row[i] = plot_out[i][0]
      col[i] = plot_out[i][1]
    plt.plot(col, row)
    plt.ylabel('Rows')
    plt.xlabel('Columns')
    plt.gca().set_xlim([-1, X_LEN])
    plt.gca().set_ylim([-1, Y_LEN])
    plt.plot(X_START, Y_START, 'ro', label='Start')
    plt.plot(X_GOAL, Y_GOAL, 'go', label='Goal')
    plt.gca().invert_yaxis()
    plt.title('The best path taken - the best policy')
    plt.legend()
    plt.show()
    print("")
    print("Starting point")
    print(X_START)
    print(Y_START)
    print("Goal point")
    print(X_GOAL)
    print(Y_GOAL)

    print("Means of first 50:")
    print(np.mean(episode_rewards[0:50]))
    print("Means of first 50:")
    print(np.mean(episode_rewards[50:100]))
    print("Means of entire rewards per episode: ")
    print(np.mean(episode_rewards))

class GridWorld:
  # Setting up the grid world
    def __init__(self):
        # Set the starting location, the current location and goal location at 0,0
        self.start_state = Y_START,X_START
        self.goal_state = Y_GOAL,X_GOAL
        #Possible positions
        self.action = ['up','right','down','left']
        #size of the grid
        self.X_LEN = X_LEN
        self.Y_LEN = Y_LEN

        #Location of the start state
        self.X_START = X_START
        self.Y_START = Y_START

        #Location of the goal state
        self.X_GOAL = X_GOAL
        self.Y_GOAL = Y_GOAL

        self.grid = np.zeros((self.Y_LEN, self.X_LEN))

    def go_to_next_loc(self, action, current_state):
      #This calculates the next location with one wind vector and four actions
      action = self.action[action]
      row = current_state[0]
      col = current_state[1]
      if action == 'up':
        new_state = max(row - 1, 0), col
      elif action == 'right':
        new_state = row,min(col + 1, self.X_LEN -1)
      elif action == 'down':
        new_state = min(row + 1, self.Y_LEN - 1),col
      else:
        new_state = row, max(col - 1,0)
      self.grid = np.zeros((self.Y_LEN, self.X_LEN))
      self.grid[new_state[0], new_state[1]] = 1
      return new_state

    def get_reward(self, loc):
      # Get the reward (-1 until you reach the goal))
      if loc != self.goal_state:
        return -1 
      else:
        return 1

#Set Hyperparameters
EPSILON = 1
GAMMA =  0.99
ALPHA = 0.5
LAMBDA = 0.9
EPISODES = 100
NUM_ACTIONS = 4
NUM_STATES = 4
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LEARNING_RATE_RMSprop = 0.0001
UPDATE_RATE = 10 
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.98 
MOMENTUM = .7
DISCOUNT_FACTOR = GAMMA
TD_DECAY = 1

# DQN algorithm for Gridworld environment
class DQN:
    def __init__(self):
        #Set all the parameters for the particular algorithm
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.alpha = ALPHA
        self.q = np.zeros((Y_LEN, X_LEN, NUM_ACTIONS))
        self.batch_size = BATCH_SIZE

        #initalize environement
        self.state_size = 1
        self.action_size = NUM_ACTIONS
        self.grid = GridWorld()

        # Initialize Replay Buffer as python deque
        self.replay_buffer = deque(maxlen=40000)

        # Set algorithm hyperparameters
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.update_rate = UPDATE_RATE
        self.momentum = MOMENTUM
        self.discount_factor = DISCOUNT_FACTOR

        # Create both Main and Target Neural Networks
        self.main_network = self.create_nn(400)
        self.target_network = self.create_nn(400)

        # Initialize Target Network with Main Network's weights
        self.target_network.set_weights(self.main_network.get_weights())

        #memory
        self.capacity = 100000
        self.list_prev_states = []
        self.list_states = []
        self.list_actions = []
        self.list_rewards = []
        self.list_bTrial_over = []

    def create_nn(self,states):
        # create DNN network for DQN with epsilon = 0.001
        model = Sequential()

        model.add(Dense(128, activation='relu', input_dim=self.state_size))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=RMSprop(learning_rate=LEARNING_RATE_RMSprop,momentum=MOMENTUM,epsilon=.001))

        return model

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.set_weights(self.main_network.get_weights())

    #memory
    def save_experience(self,prev_state, action, reward, state, bTrial_over): #prev state is state and next state is now state

        self.list_prev_states.append(prev_state)
        self.list_states.append(state)
        self.list_rewards.append(reward)
        self.list_bTrial_over.append(bTrial_over)
        self.list_actions.append(action)

        if(self.list_rewards.__len__() > self.capacity):
            del self.list_prev_states[0]
            del self.list_states[0]
            del self.list_actions[0]
            del self.list_rewards[0]
            del self.list_bTrial_over[0]

    def sample_experience_batch(self):
        experience_indices = np.random.randint(0, self.list_rewards.__len__(), self.batch_size)
        prev_states = []
        actions = []
        rewards = []
        states = []
        bTrial_over = []

        for i in experience_indices:

            prev_states.append(self.list_prev_states[i])
            actions.append(self.list_actions[i])
            rewards.append(self.list_rewards[i])
            states.append(self.list_states[i])
            bTrial_over.append(self.list_bTrial_over[i])

        state_batch = np.squeeze(np.array(prev_states, dtype=int))
        action_batch = np.array(actions, dtype=int)
        reward_batch = np.array(rewards, dtype=float)
        next_state_batch = np.squeeze(np.array(states, dtype=int))
        terminal_batch = bTrial_over
        # Return a tuple, where each item corresponds to each array/batch created above
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def reshape_state(self,state):
      mat = []
      for i in range(self.batch_size):
        mat.append([state])
      return mat

    def reshape(self, state):
        new_state = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            new_state[i] = state

    def pick_epsilon_greedy_action(self,state):
        # Pick random action with probability ε
        curr_loc = self.convert_state_back(state)
        if random.uniform(0, 1) < self.epsilon: 
            ca = np.random.randint(self.action_size)
            return ca
        else:
          state_batch, _, _, _, _ = self.sample_experience_batch()
          state_batch[0] = state
          q_values = self.main_network.predict(state_batch, verbose=0)
          return np.argmax(q_values[0])

    def train(self):
        # Sample a batch of experiences
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.sample_experience_batch()

        # Get the actions with highest Q-Value for the batch of next states
        next_q = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q, axis=1)
        # Get the Q-Values of each state in the batch of states
        q_values = self.main_network.predict(state_batch, verbose=0)

        # Update the Q-Value corresponding to the current action with the Target Value
        for i in range(self.batch_size):
            q_values[i][action_batch[i]] = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]
     
        # Fit the Neural Network
        self.main_network.fit(state_batch, q_values, verbose=0)

    def convert_state(self,location):
      state = 0
      count = 1
      for i in range(Y_LEN):
        for j in range(X_LEN):
          if i == location[0] and j == location[1]:
            state = count
          count += 1
      return state
    def convert_state_back(self,state):
      state_idx = 0
      location_x = 0
      location_y = 0
      count = 1
      for i in range(Y_LEN):
        for j in range(X_LEN):
          if count == state:
            location_x = j 
            location_y = i
          count += 1
      return location_y,location_x

#This code was implemented by us
    def run_gridworld(self):
        #initialize the way to store the rewards
        episode_rewards = np.zeros(EPISODES)
        paths_taken = []
        time_steps = 0
        episodes_num = []
        time_step_episode = []
        for episode in range (EPISODES):
          accum_rewards = 0
          notEndReached = True
          path_taken = []
          #Initialize the state
          current_state = self.convert_state(self.grid.start_state)  
          notEndReached = True
          #initialize neural network sequence
          terminal_state_reached = False
          time_steps_episode = 0
          ogtimesteps = time_steps
          print(f'\nTraining on EPISODE {episode+1} with epsilon {self.epsilon}')
          while notEndReached and time_steps_episode < 1000:
            time_steps = time_steps + 1
            time_steps_episode += 1
            if time_steps % self.update_rate == 0:
                self.update_target_network()
            #choose the action
            chosen_action = self.pick_epsilon_greedy_action(current_state)
            #Take action A and observe the reqard and the state
            next_state = self.convert_state(self.grid.go_to_next_loc(chosen_action, self.convert_state_back(current_state)))
            #Choose action A ( the next location to move to) using greedy
            
            #check if terminal
            if current_state == self.convert_state(self.grid.goal_state):
              terminal_state_reached = True
            #get reward
            curr_reward = self.grid.get_reward(self.convert_state_back(current_state))

            #save the experience
            self.save_experience(current_state, chosen_action, curr_reward, next_state, terminal_state_reached)
            if len(self.list_actions) > self.batch_size:
                self.train()
            accum_rewards =  accum_rewards + self.grid.get_reward(self.convert_state_back(current_state))
            path_taken.append(self.convert_state_back(current_state))
            if current_state == self.convert_state(self.grid.goal_state):
              print('Episode: ', episode+1, ',' ' terminated with Reward ', accum_rewards)
              notEndReached = False
            #Reset the current state
            current_state = next_state
          print(f'Time elapsed during EPISODE {episode+1}: {time_steps - ogtimesteps} steps ')

          self.epsilon = self.epsilon * self.epsilon_decay
          #Add the parameters of the episode
          episodes_num.append(episode)

          time_step_episode.append(time_steps)
          episode_rewards[episode] = accum_rewards
          paths_taken.append(path_taken)
        #Print the details
        print_out_everything(episode_rewards,time_step_episode, episodes_num, paths_taken )


# SOM for Gridworld environment
class SOM(object):

    def __init__(self,  maze_width, maze_height, input_dim, map_size, learning_rate, sigma, sigma_const):
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.SOM_layer = SOMLayer(np.amax([maze_width, maze_height]), input_dim, map_size, learning_rate, sigma, sigma_const)
        self.location_counts = np.zeros((maze_height, maze_width))

    def Update(self, state, best_unit, reward_value):
        self.SOM_layer.Update(state, best_unit, reward_value)

    def GetOutput(self, state):
        best_unit = self.SOM_layer.GetBestUnit(state)
        return best_unit

    def PlotResults(self, plot_num):
        self.PlotMap()
        self.PlotLocations()

    def PlotMap(self):
        width = np.unique(self.SOM_layer.units['xy']).shape[0]
        height = width
        im_grid = np.zeros((width, height, 3))
        for i in range(width * height):
            image = np.zeros(3)
            image[:2] = self.SOM_layer.units['w'][i, :]
            image = np.clip(np.array(image) / np.amax([self.maze_width, self.maze_height]), 0, 1)
            im_grid[self.SOM_layer.units['xy'][i, 0], self.SOM_layer.units['xy'][i, 1], :] = image
        plt.figure()
        plt.imshow(im_grid)
        plt.close()

    def PlotLocations(self):
        im_grid = np.zeros((self.maze_height, self.maze_width))

        for i in range(self.SOM_layer.num_units):
            y = int(np.rint(np.clip(self.SOM_layer.units['w'][i, 0], 0, self.maze_height-1)))
            x = int(np.rint(np.clip(self.SOM_layer.units['w'][i, 1], 0, self.maze_width-1)))
            im_grid[y, x] = 1
        plt.figure()
        plt.imshow(im_grid)
        plt.close()
        np.save(self.directory + 'SOMLocations', im_grid)

    def RecordLocationCounts(self):
        for i in range(self.SOM_layer.num_units):
            y = int(np.clip(self.SOM_layer.units['w'][i, 0], 0, self.maze_height-1))
            x = int(np.clip(self.SOM_layer.units['w'][i, 1], 0, self.maze_width-1))
            self.location_counts[y, x] += 1

class SOMLayer:
  def __init__(self, grid_dim, input_dim, size, learning_rate, sigma, sigma_const):
      self.size = size
      self.num_units = size * size
      self.num_dims = input_dim
      self.num_weights = input_dim

      self.learning_rate = learning_rate
      self.sigma = sigma
      self.sigma_const = sigma_const

      self.units = {'xy': [], 'w': []}
      self.ConstructMap(grid_dim)

  def ConstructMap(self, maze_dim):
        x = 0
        y = 0
        # Construct map
        for u in range(self.num_units):

            self.units['xy'].append([x, y])
            self.units['w'].append(np.random.rand(self.num_weights) * maze_dim)

            x += 1
            if (x >= self.size):
                x = 0
                y += 1

        self.units['xy'] = np.array(self.units['xy'])
        self.units['w'] = np.array(self.units['w'])

  def Update(self, state, best_unit, reward_value):
      diffs = self.units['xy'] - self.units['xy'][best_unit, :]
      location_distances = np.sqrt(np.sum(np.square(diffs), axis=-1))
      neighbourhood_values = np.exp(-np.square(location_distances) / (
                2.0 * (self.sigma_const + (reward_value * self.sigma))))
      
      self.units['w'] += (reward_value * self.learning_rate) * \
                           np.expand_dims(neighbourhood_values, axis=-1) * (state - self.units['w'])

  def GetBestUnit(self, state):
      best_unit = np.argmin(np.sum((self.units['w'] - state) ** 2, axis=-1), axis=0)
      return best_unit


from threading import activeCount
class CTDL:
    def __init__(self):
        #Set all the parameters for the particular algorithm
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.alpha = ALPHA
        self.q = np.zeros((Y_LEN, X_LEN, NUM_ACTIONS))

        self.batch_size = BATCH_SIZE

        self.weighting_decay = 10
        self.som_size = 8 
        self.som_alpha = 0.01
        self.som_sigma = .1 
        self.som_sigma_const = .1

        #initalize environement
        self.state_size = 1
        self.action_size = NUM_ACTIONS
        self.grid = GridWorld()

        # Initialize Replay Buffer as python deque
        self.replay_buffer = deque(maxlen=40000)

        # Set algorithm hyperparameters
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = 0.98
        self.learning_rate = LEARNING_RATE
        self.momentum = MOMENTUM
        self.update_rate = UPDATE_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.TD_decay = TD_DECAY

        # Create both Main and Target Neural Networks
        self.main_network = self.create_nn(400)
        self.target_network = self.create_nn(400)

        # Initialize Target Network with Main Network's weights
        self.target_network.set_weights(self.main_network.get_weights())

         #memory
        self.capacity = 100000
        self.list_prev_states = []
        self.list_states = []
        self.list_actions = []
        self.list_rewards = []
        self.list_bTrial_over = []

        #SOM
        self.som = self.createSOM(self.som_alpha, self.som_sigma, self.som_sigma_const) #took away som_size and replaced with batch size

    def createSOM(self, som_alpha, som_sigma, som_sigma_const):
        self.SOM = SOM( self.grid.X_LEN, self.grid.Y_LEN, 2, self.som_size,
                       som_alpha, som_sigma,
                       som_sigma_const)
        self.Q_alpha = .9
        self.QValues = np.zeros((self.som_size * self.som_size, NUM_ACTIONS))


    def create_nn(self,states):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=self.state_size))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=RMSprop(learning_rate=LEARNING_RATE_RMSprop,momentum=MOMENTUM,epsilon=.001))

        return model

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.set_weights(self.main_network.get_weights())

    #memory
    def save_experience(self,prev_state, action, reward, state, bTrial_over): #prev state is state and next state is now state

        self.list_prev_states.append(prev_state)
        self.list_states.append(state)
        self.list_rewards.append(reward)
        self.list_bTrial_over.append(bTrial_over)
        self.list_actions.append(action)

        if self.list_rewards.__len__() > self.capacity:
            del self.list_prev_states[0]
            del self.list_states[0]
            del self.list_actions[0]
            del self.list_rewards[0]
            del self.list_bTrial_over[0]

    def sample_experience_batch(self):
        experience_indices = np.random.randint(0, self.list_rewards.__len__(), self.batch_size)
        prev_states = []
        actions = []
        rewards = []
        states = []
        bTrial_over = []
        for i in experience_indices:

            prev_states.append(self.list_prev_states[i])
            actions.append(self.list_actions[i])
            rewards.append(self.list_rewards[i])
            states.append(self.list_states[i])
            bTrial_over.append(self.list_bTrial_over[i])
        state_batch = np.squeeze(np.array(prev_states, dtype=int))
        action_batch = np.array(actions, dtype=int)
        reward_batch = np.array(rewards, dtype=float)
        next_state_batch = np.squeeze(np.array(states, dtype=int))
        terminal_batch = bTrial_over
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def reshape_state(self,location):
      mat = np.zeros(100)
      count = -1
      for i in range(Y_LEN):
        for j in range(X_LEN):
          count += 1
          mat[count] = 0
          if i == location[0] and j == location[1]:
            mat[count] = 1
      return mat

    def get_q_values(self, input):
      q_values_DNN = self.main_network.predict(input, verbose=0)
      q_values = np.zeros((len(q_values_DNN),4))
      count = 0
      for sta in input:
        best_unit_som = self.SOM.GetOutput(self.convert_state_back(sta)) #action
        som_action_values = self.QValues[best_unit_som, :]
        w = self.GetWeighting(best_unit_som, sta)
        q_vals = ((w * som_action_values) + ((1 - w) * q_values_DNN[count]))
        q_values[count,:] = q_vals
        count += 1
      return q_values

    def pick_epsilon_greedy_action(self,state):
        # Pick random action with probability ε
        if random.uniform(0, 1) < self.epsilon: 
            ca = np.random.randint(self.action_size)
            return ca
        else:
          state_batch, _, _, _, _ = self.sample_experience_batch()
          state_batch[0] = state
          q_values = self.get_q_values(state_batch)
          return np.argmax(q_values[0])

    def train(self):
        # Sample a batch of experiencesh
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.sample_experience_batch()

        # Get the actions with highest Q-Value for the batch of next states
        next_q = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q, axis=1)
        # Get the Q-Values of each state in the batch of states
        
        q_values = self.get_q_values(state_batch)

        # Update the Q-Value corresponding to the current action with the Target Value
        for i in range(self.batch_size):
          q_values[i][action_batch[i]] = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i] 
        # Fit the Neural Network
        self.main_network.fit(state_batch, q_values, verbose=0)

    def convert_state(self,location):
      state = 0
      count = 1
      for i in range(Y_LEN):
        for j in range(X_LEN):
          if i == location[0] and j == location[1]:
            state = count
          count += 1
      return state
    
    def convert_state_back(self,state):
      state_idx = 0
      location_x = 0
      location_y = 0
      count = 1
      for i in range(Y_LEN):
        for j in range(X_LEN):
          if count == state:
            location_x = j 
            location_y = i
          count += 1
      return location_y,location_x

    def UpdateSOM(self, target,prev_state,prev_action):
        prev_best_unit = self.SOM.GetOutput(prev_state)
        state_batch, _, _, _, _ = self.sample_experience_batch()
        state_batch[0] = prev_state
        action = self.main_network.predict(state_batch, verbose=0)
        delta = np.exp(np.abs(target -
                              action[0][prev_action]) / self.TD_decay) - 1

        delta = np.clip(delta, 0, 1)
        self.SOM.Update(prev_state, prev_best_unit, delta)

        prev_best_unit = self.SOM.GetOutput(prev_state)
        w = self.GetWeighting(prev_best_unit, prev_state)
        self.QValues[prev_best_unit, prev_action] += self.Q_alpha * w * (target - self.QValues[prev_best_unit, prev_action])
        self.SOM.RecordLocationCounts()
        self.train()
        return

    def GetTargetValue(self, bTrial_over, reward, state):
        state_batch, _, _, _, _ = self.sample_experience_batch()
        state_batch[0] = state
        qvalues = self.get_q_values(state_batch) 
        q_values = qvalues[0]
        max_q_value = np.amax(q_values)
        if bTrial_over:
            target = reward
        else:
            target = reward + (max_q_value * self.discount_factor)
        return target

    def GetWeighting(self, best_unit, state):

        diff = np.sum(np.square(self.SOM.SOM_layer.units['w'][best_unit, :] - state))
        w = np.exp(-diff / self.weighting_decay)

        return w

#This code was implemented by us
    def run_gridworld(self):
        #initialize the way to store the rewards
        episode_rewards = np.zeros(EPISODES)
        paths_taken = []
        time_steps = 0
        episodes_num = []
        time_step_episode = []
        for episode in range(EPISODES):
          accum_rewards = 0
          notEndReached = True
          path_taken = []
          #Initialize the state
          current_state = self.convert_state(self.grid.start_state)  
          notEndReached = True
          #initialize neural network sequence
          terminal_state_reached = False
          time_steps_episode = 0
          print(f'\nTraining on EPISODE {episode+1} with epsilon {self.epsilon}')
          ogtimesteps = time_steps
          while notEndReached and time_steps_episode < 1000:
            time_steps = time_steps + 1
            time_steps_episode += 1
            if time_steps % self.update_rate == 0: 
                self.update_target_network()
            #choose the action
            chosen_action = self.pick_epsilon_greedy_action(current_state)
            #Take action A and observe the reqard and the state
            next_state = self.convert_state(self.grid.go_to_next_loc(chosen_action, self.convert_state_back(current_state)))
            #Choose action A ( the next location to move to) using greedy
            #check if terminal
            if current_state == self.convert_state(self.grid.goal_state):
              terminal_state_reached = True
            #get reward
            curr_reward = self.grid.get_reward(self.convert_state_back(current_state))
            #save the experience
            self.save_experience(current_state, chosen_action, curr_reward, next_state, terminal_state_reached)
            target = self.GetTargetValue(terminal_state_reached, curr_reward, current_state)
            self.prev_action = chosen_action
            self.prev_state = current_state
            current_state = next_state
            
           #Record the parameters of this loop
            if len(self.list_rewards) > self.batch_size:
              self.train()
              self.UpdateSOM(target,current_state,chosen_action)
            accum_rewards =  accum_rewards + self.grid.get_reward(self.convert_state_back(current_state))
            path_taken.append(self.convert_state_back(current_state))
            if current_state == self.convert_state(self.grid.goal_state):
              print('Episode: ', episode+1, ',' ' terminated with Reward ', accum_rewards)
              notEndReached = False
            #Reset the current state
          print(f'Time elapsed during EPISODE {episode+1}: {time_steps - ogtimesteps} step')

          self.epsilon = self.epsilon * self.epsilon_decay
          #Add the parameters of the episode
          episodes_num.append(episode)
          time_step_episode.append(time_steps)
          episode_rewards[episode] = accum_rewards
          paths_taken.append(path_taken)
        #Print the details
        print_out_everything(episode_rewards,time_step_episode, episodes_num, paths_taken)

#dnn=DQN()
#dnn.run_gridworld()

ctdl=CTDL()
ctdl.run_gridworld()

