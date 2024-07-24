import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

# Define the state and action space
state_size = 10  # Example state size
action_size = 5  # Example action size

# Initialize DQNs for power allocation and radio resource allocation
dqn_power = DQN(state_size, action_size)
dqn_radio = DQN(state_size, action_size)

# Example environment interaction
for episode in range(1000):
    state = np.random.rand(1, state_size)  # Example initial state
    epsilon = max(0.01, 0.1 - episode / 500)  # Decaying epsilon for exploration
    done = False

    while not done:
        action_power = dqn_power.act(state, epsilon)
        next_state = np.random.rand(1, state_size)  # Example state transition
        reward = np.random.rand()  # Example reward
        done = np.random.rand() < 0.1  # Example terminal state condition

        dqn_power.train(state, action_power, reward, next_state, done)
        state = next_state

        # Interact with radio resource allocation DQN
        action_radio = dqn_radio.act(state, epsilon)
        next_state_radio = np.random.rand(1, state_size)  # Example state transition
        reward_radio = np.random.rand()  # Example reward
        done_radio = np.random.rand() < 0.1  # Example terminal state condition

        dqn_radio.train(state, action_radio, reward_radio, next_state_radio, done_radio)
        state = next_state_radio

        if done or done_radio:
            break