# main.py
from env import MarathonEnv
from agent import DQNAgent
import numpy as np

class TrainingEnvironment(MarathonEnv):
    def __init__(self):
        super().__init__()
        # Actions discrètes disponibles
        self.actions = [
        {'type': 'rest', 'volume': 0, 'intensity': 0},     # Plus de repos
        {'type': 'rest', 'volume': 0, 'intensity': 0},
        {'type': 'easy', 'volume': 6, 'intensity': 0.6},   # Sessions plus légères
        {'type': 'easy', 'volume': 8, 'intensity': 0.6},
        {'type': 'easy', 'volume': 10, 'intensity': 0.6},
        {'type': 'tempo', 'volume': 6, 'intensity': 0.7},  # Intensité réduite
        {'type': 'tempo', 'volume': 8, 'intensity': 0.7},
        {'type': 'intervals', 'volume': 4, 'intensity': 0.8},
        {'type': 'long_run', 'volume': 12, 'intensity': 0.6},
        {'type': 'long_run', 'volume': 15, 'intensity': 0.6}
    ]

    def get_state_vector(self):
        return np.array([
            self.state['fitness'],
            self.state['fatigue'],
            self.state['days_to_goal'] / 90,  # normalisation
            self.state['last_week_volume'] / 100,  # normalisation
            self.state['performance'],
            self.state['form']
        ])

def train():
    env = TrainingEnvironment()
    agent = DQNAgent(state_size=6, action_size=10)
    episodes = 500
    
    for e in range(episodes):
        env = TrainingEnvironment()
        state = env.get_state_vector()
        total_reward = 0
        
        for time in range(84):
            action_index = agent.act(state)
            action = env.actions[action_index]
            
            next_state, reward, done, _ = env.step(action)
            next_state = env.get_state_vector()
            
            # Stockage des expériences
            agent.remember(state, action_index, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
                
            if len(agent.memory) > agent.batch_size:
                agent.replay()
                
        if e % 10 == 0:
            print(f"episode: {e}/{episodes}, score: {total_reward}, epsilon: {agent.epsilon:.2}")
if __name__ == "__main__":
    train() 