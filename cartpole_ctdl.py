# ***************************************************************************************/
# *    Title: CTDL source code
# *    Author:Sam Blakeman, Denis Mareschal
# *    Date: 2020
# *    Availability:https://github.com/SamBlakeman/CTDL.git
# *
# ***************************************************************************************/

import random
import time
from collections import deque

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def print_out_everything(episode_rewards,time_step_episode, episodes_num, paths_taken):
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

class DeepSOM(object):

    def __init__(self,input_dim, map_size, learning_rate, sigma, sigma_const):
        self.SOM_layer = SOMLayer(input_dim, map_size, learning_rate, sigma, sigma_const)
        return

    def Update(self, state, best_unit, reward_value):
        self.SOM_layer.Update(state, best_unit, reward_value)
        return

    def GetOutput(self, state):
        best_unit = self.SOM_layer.GetBestUnit(state)
        return best_unit
class SOMLayer():

    def __init__(self, input_dim, size, learning_rate, sigma, sigma_const):
        self.size = size
        self.num_units = size * size
        self.num_weights = input_dim

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.sigma_const = sigma_const

        self.units = {'xy': [], 'w': []}
        self.ConstructMap()
        return

    def ConstructMap(self):
        x = 0
        y = 0

        # Construct map
        for _ in range(self.num_units):
            self.units['xy'].append([x, y])
            self.units['w'].append(np.random.rand(self.num_weights))
            x += 1

            if x >= self.size:
                x = 0
                y += 1

        self.units['xy'] = np.array(self.units['xy'])
        self.units['w'] = np.array(self.units['w'])
        return

    def Update(self, state, best_unit, reward_value):

        diffs = self.units['xy'] - self.units['xy'][best_unit, :]
        location_distances = np.sqrt(np.sum(np.square(diffs), axis=-1))
        neighbourhood_values = np.exp(-np.square(location_distances) / (2.0 * (self.sigma_const + (reward_value * self.sigma))))

        num = (reward_value * self.learning_rate) * \
                           np.expand_dims(neighbourhood_values, axis=-1) * (state - self.units['w'])[0]
        num2 = (self.units['w'] + num)[0]
        self.units['w'] = num2
        return

    def GetBestUnit(self, state):

        best_unit = np.argmin(np.sum((self.units['w'] - state) ** 2, axis=-1), axis=0)
        return best_unit

class CTDLAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # Initialize Replay Buffer as python deque
        self.replay_buffer = deque(maxlen=40000)

        # Set algorithm hyperparameters
        self.TD_decay = 1
        self.gamma = 0.99
        self.epsilon = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_rate = 10
        self.discount_factor = 0.99

        self.weighting_decay = 10
        self.som_size = 15
        self.som_alpha = 0.1
        self.som_sigma = .06
        self.som_sigma_const = .1
        self.input_dim = state_size

        # Create both Main and Target Neural Networks
        self.main_network = self.create_nn()
        self.target_network = self.create_nn()

        # Initialize Target Network with Main Network's weights
        self.target_network.set_weights(self.main_network.get_weights())
        self.CreateSOM( self.som_size,self.som_alpha, self.som_sigma, self.som_sigma_const) #took away som_size and replaced with batch size

    def CreateSOM(self, som_size,som_alpha, som_sigma, som_sigma_const):

        self.SOM = DeepSOM(self.input_dim, som_size,
                           som_alpha, som_sigma,
                           som_sigma_const)
        self.Q_alpha = som_alpha
        self.QValues = np.zeros((som_size * som_size, self.action_size))
        return

    def GetWeighting(self, best_unit, state):
        diff = np.sum(np.square(self.SOM.SOM_layer.units['w'][best_unit] - state))
        w = np.exp(-diff / self.weighting_decay)
        return w
    
    def create_nn(self):
        model = Sequential()

        model.add(Dense(32, activation='relu', input_dim=self.state_size))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.set_weights(self.main_network.get_weights())

    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def sample_experience_batch(self, batch_size):
        # Sample {batchsize} experiences from the Replay Buffer
        exp_batch = random.sample(self.replay_buffer, batch_size)

        # Create an array with the {batchsize} elements for s, a, r, s' and terminal information
        state_batch = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size)
        terminal_batch = [batch[4] for batch in exp_batch]

        # Return a tuple, where each item corresponds to each array/batch created above
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def get_q_values(self, input):
        q_values_DNN = self.main_network.predict(input, verbose=0)
        q_values = np.zeros((len(q_values_DNN),self.action_size))
        count = 0
        for state in input:
            best_unit = self.SOM.GetOutput(state)
            som_action_values = self.QValues[best_unit, :]
            w = self.GetWeighting(best_unit, state)
            q_vals = (w * som_action_values[0]) + ((1 - w) * q_values_DNN[0])
            q_values[count,:] = q_vals
            count += 1
        return q_values

    def pick_epsilon_greedy_action(self, state):

        # Pick random action with probability ε
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        # Pick action with highest Q-Value (item with highest value for Main NN's output)
        state = state.reshape((1, self.state_size))
        q_values = self.get_q_values(state)
        return np.argmax(q_values[0])

    def train(self, batch_size):

        # Sample a batch of experiences
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.sample_experience_batch(batch_size)

        # Get the actions with highest Q-Value for the batch of next states
        next_q = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q, axis=1)
        # Get the Q-Values of each state in the batch of states
        q_values = self.get_q_values(state_batch) 

        # Update the Q-Value corresponding to the current action with the Target Value
        for i in range(batch_size):
            q_values[i][action_batch[i]] = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]

        # Fit the Neural Network
        self.main_network.fit(state_batch, q_values, verbose=0)

    def UpdateSOM(self,target):
        prev_best_unit = self.SOM.GetOutput(self.prev_state)
        state = self.prev_state.reshape((1, self.state_size))
        q_values = self.main_network.predict(state, verbose=0)
        delta = np.exp(np.abs(target -
                              q_values[0][self.prev_action]) / self.TD_decay) - 1

        delta = np.clip(delta, 0, 1)
        self.SOM.Update(self.prev_state, prev_best_unit, delta)

        prev_best_unit = self.SOM.GetOutput(self.prev_state)
        w = self.GetWeighting(prev_best_unit, self.prev_state)
        self.QValues[prev_best_unit, self.prev_action] += self.Q_alpha * w * (target - self.QValues[prev_best_unit, self.prev_action])
        self.train(self.batch_size)
        return

    def GetTargetValue(self, bTrial_over, reward, state):

        state = state.reshape((1, self.state_size))
        q_values = self.get_q_values(state)
        max_q_value = np.amax(q_values)
        if bTrial_over:
            target = reward
        else:
            target = reward + (max_q_value * self.discount_factor)
        return target

if __name__ == '__main__':

    # Initialize CartPole environment
    env = gym.make("CartPole-v1")
    state, _ = env.reset()

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    ctdl_agent = CTDLAgent(state_size, action_size)
    ctdl_agent.state_size = state_size
    ctdl_agent.action_size = action_size
    # Define number of episodes, timesteps per episode and batch size
    num_episodes = 100
    num_timesteps = 500
    ctdl_agent.batch_size = 64
    time_step = 0  # Initialize timestep counter, used for updating Target Network
    rewards, epsilon_values = list(), list()  # List to keep logs of rewards and epsilon values, for plotting later
    paths_taken = []
    time_steps = 0
    episodes_num = []
    time_step_episode = []
    for ep in range(num_episodes):
        tot_reward = 0
        state, _ = env.reset()
        print(f'\nTraining on EPISODE {ep+1} with epsilon {ctdl_agent.epsilon}')
        start = time.time()
        path_taken = []
        for t in range(num_timesteps):
            time_step += 1

            # Update Target Network every {dqn_agent.update_rate} timesteps
            if time_step % ctdl_agent.update_rate == 0:
                ctdl_agent.update_target_network()

            action = ctdl_agent.pick_epsilon_greedy_action(state)  # Select action with ε-greedy policy
            next_state, reward, terminal, _, _ = env.step(action)  # Perform action on environment
            ctdl_agent.save_experience(state, action, reward, next_state, terminal)  # Save experience in Replay Buffer

            # Update current state to next state and total reward
            target = ctdl_agent.GetTargetValue(terminal, reward, state)
            ctdl_agent.prev_action = action
            ctdl_agent.prev_state = state
            state = next_state
            tot_reward += reward

            if terminal:
                print('Episode: ', ep+1, ',' ' terminated with Reward ', tot_reward)
                break

            # Train the Main NN when ReplayBuffer has enough experiences to fill a batch
            if len(ctdl_agent.replay_buffer) > ctdl_agent.batch_size:
                ctdl_agent.train(ctdl_agent.batch_size)
                ctdl_agent.UpdateSOM(target)
        rewards.append(tot_reward)
        epsilon_values.append(ctdl_agent.epsilon)

        # Everytime an episode ends, update Epsilon value to a lower value
        if ctdl_agent.epsilon > ctdl_agent.epsilon_min:
            ctdl_agent.epsilon *= ctdl_agent.epsilon_decay

        # Print info about the episode performed
        elapsed = time.time() - start
        print(f'Time elapsed during EPISODE {ep+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')

        # If the agent got a reward >499 in each of the last 10 episodes, the training is terminated
        if sum(rewards[-10:]) > 4990:
            print('Training stopped because agent has performed a perfect episode in the last 10 episodes')
            break
        episodes_num.append(ep)
        time_step_episode.append(time_step)
        paths_taken.append(path_taken)

    #Print the details
    print_out_everything(rewards,time_step_episode, episodes_num, paths_taken)