# agent.py
import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size=6, action_size=10):
        self.state_size = state_size      # fitness, fatigue, days_to_goal, last_week_volume, performance, form
        self.action_size = action_size    # nombre d'actions discrètes
        
        # Hyperparamètres
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # taux d'exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.01
        self.batch_size = 32
        
        # Modèles
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Reshape des states
                state = np.reshape(state, [1, self.state_size])
                next_state = np.reshape(next_state, [1, self.state_size])
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            state = np.reshape(state, [1, self.state_size])  # Important!
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Reshape de state
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])