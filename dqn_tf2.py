import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.optimizers.legacy import Adam  # type: ignore
from tensorflow.keras.optimizers.legacy import SGD # type: ignore
from tensorflow.keras.models import load_model # type: ignore

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), 
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        #self.state_memory[index] = np.array(state)
        #print("state_memory type",np.shape(self.state_memory), type(self.state_memory))
        self.state_memory[index] = state

        #print("state_memory type",np.shape(self.state_memory), type(self.state_memory))
        #print(self.state_memory)
        #self.state_memory = np.delete(self.state_memory,(0,0))

        #print("state_memory type",np.shape(self.state_memory), type(self.state_memory))
        #print(self.state_memory)
        #self.new_state_memory[index] = state_
        #self.new_state_memory = np.delete(self.new_state_memory,0)
        print("reward=",reward)
        
        self.reward_memory[index] = reward
        #self.reward_memory = np.delete(self.reward_memory,0)
        self.action_memory[index] = action
        #self.action_memory = np.delete(self.action_memory,0)
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=0.002475, epsilon_end=0.01,
                mem_size=64000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)
        self.episode_count = 0
    
    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)

        return action


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)


        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones


        self.q_eval.train_on_batch(states, q_target)

        
        # if self.epsilon < 0.01:
        if self.episode_count < 1500:
            #self.epsilon = 0
            self.epsilon = 1
        elif self.episode_count > 1500 and self.epsilon > 0.01:
            self.epsilon = self.epsilon - self.eps_dec
        elif self.episode_count > 1500 and self.epsilon <= 0.01:
            #self.epsilon = 0
            self.epsilon = 0.01
        
        #self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
        #        self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file)
        
    def episode_counter_in(self, episode_counter):
        self.episode_count = episode_counter