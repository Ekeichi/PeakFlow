import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class MarathonEnv:
    def __init__(self):
        self.state = {
            'fitness': 0.2,
            'fatigue': 0.0,
            'days_to_goal': 90,
            'last_week_volume': 20,
            'performance': 0.1,
            'form': 0.0
        }
        self.tau_fatigue = 15
        self.tau_fitness = 45
        self.history = {
            'fitness': [self.state['fitness']],
            'fatigue': [self.state['fatigue']],
            'performance': [self.state['performance']],
            'form': [self.state['form']]
        }
        
    def step(self, action):
        if not self._is_safe(action):
            print("⚠️ Action non sécurisée")
            return self.state, -10, True, {"error": "unsafe_action"}
            
        new_state = self._simulate_effects(action)
        reward = self._calculate_reward(new_state)
        done = new_state['days_to_goal'] <= 0
        
        # Mettre à jour l'historique
        self.history['fitness'].append(new_state['fitness'])
        self.history['fatigue'].append(new_state['fatigue'])
        self.history['performance'].append(new_state['performance'])
        self.history['form'].append(new_state['form'])
        
        self.state = new_state
        return new_state, reward, done, {}
        
    def _is_safe(self, action):
        if action['type'] == 'rest':
            return True
            
        if self.state['fatigue'] > 0.8 and action['intensity'] > 0.7:
            print("⚠️ Fatigue trop élevée pour séance intense")
            return False
            
        if action['volume'] > self.state['last_week_volume'] * 1.2:
            print("⚠️ Augmentation volume trop importante")
            return False
            
        if self.state['form'] < -0.3 and action['intensity'] > 0.7:
            print("⚠️ Forme trop basse pour séance intense")
            return False
            
        return True
        
    def _simulate_effects(self, action):
        new_state = deepcopy(self.state)
        
        effort = (action['volume'] * action['intensity']) / 100
        
        if action['type'] == 'rest':
            new_state['fatigue'] = self.state['fatigue'] * np.exp(-1/self.tau_fatigue)
            new_state['fitness'] = self.state['fitness'] * np.exp(-1/self.tau_fitness)
        else:
            new_state['fatigue'] = (effort + np.exp(-1/self.tau_fatigue) * self.state['fatigue'])
            new_state['fitness'] = (effort + np.exp(-1/self.tau_fitness) * self.state['fitness'])
            
        new_state['performance'] = (new_state['fitness'] - new_state['fatigue'])/2
        if new_state['performance'] < 0.05:
            new_state['performance'] = 0.05
            
        new_state['form'] = new_state['fitness'] - 2*new_state['fatigue']
        
        new_state['fatigue'] = min(1.0, max(0.0, new_state['fatigue']))
        new_state['fitness'] = min(1.0, max(0.0, new_state['fitness']))
        
        new_state['days_to_goal'] -= 1
        return new_state
        
    def _calculate_reward(self, new_state):
        reward = 0
        
        # Récompense progression performance
        perf_delta = new_state['performance'] - self.state['performance']
        reward += 50 * perf_delta
        
        # Bonus/Malus forme
        if new_state['form'] > 0.1:
            reward += 2
        elif new_state['form'] < -0.3:
            reward -= 5
            
        # Pénalité fatigue excessive
        if new_state['fatigue'] > 0.3:
            reward -= (new_state['fatigue'] - 0.3) * 10
            
        # Bonus progression fitness
        fitness_gain = new_state['fitness'] - self.state['fitness']
        if fitness_gain > 0:
            reward += 20 * fitness_gain
            
        return reward
        
    def reset(self):
        self.__init__()
        return self.state

def test_scenarios():
    """Test différents scénarios d'entraînement"""
    
    # Test 1: Semaine équilibrée
    env = MarathonEnv()
    print("\n=== Test 1: Semaine équilibrée ===")
    balanced_week = [
        {'type': 'easy', 'volume': 10, 'intensity': 0.6},
        {'type': 'rest', 'volume': 0, 'intensity': 0},
        {'type': 'tempo', 'volume': 12, 'intensity': 0.8},
        {'type': 'easy', 'volume': 8, 'intensity': 0.6},
        {'type': 'long_run', 'volume': 20, 'intensity': 0.7}
    ]
    
    for i, session in enumerate(balanced_week, 1):
        state, reward, done, _ = env.step(session)
        print(f"\nJour {i}:")
        print(f"Session: {session}")
        print(f"Fitness: {state['fitness']:.3f}")
        print(f"Fatigue: {state['fatigue']:.3f}")
        print(f"Performance: {state['performance']:.3f}")
        print(f"Forme: {state['form']:.3f}")
        print(f"Récompense: {reward:.2f}")
    
    plot_training_response(env.history)
    
    # Test 2: Semaine intensive
    env = MarathonEnv()
    print("\n=== Test 2: Semaine intensive ===")
    heavy_week = [
        {'type': 'tempo', 'volume': 15, 'intensity': 0.8},
        {'type': 'intervals', 'volume': 12, 'intensity': 0.9},
        {'type': 'tempo', 'volume': 15, 'intensity': 0.8},
        {'type': 'long_run', 'volume': 25, 'intensity': 0.7}
    ]
    
    for i, session in enumerate(heavy_week, 1):
        state, reward, done, _ = env.step(session)
        print(f"\nJour {i}:")
        print(f"Session: {session}")
        print(f"Fitness: {state['fitness']:.3f}")
        print(f"Fatigue: {state['fatigue']:.3f}")
        print(f"Performance: {state['performance']:.3f}")
        print(f"Forme: {state['form']:.3f}")
        print(f"Récompense: {reward:.2f}")
    
    plot_training_response(env.history)
    
    # Test 3: Récupération
    env = MarathonEnv()
    print("\n=== Test 3: Semaine récupération ===")
    recovery_week = [
        {'type': 'easy', 'volume': 8, 'intensity': 0.6},
        {'type': 'rest', 'volume': 0, 'intensity': 0},
        {'type': 'rest', 'volume': 0, 'intensity': 0},
        {'type': 'easy', 'volume': 6, 'intensity': 0.6},
        {'type': 'easy', 'volume': 10, 'intensity': 0.6}
    ]
    
    for i, session in enumerate(recovery_week, 1):
        state, reward, done, _ = env.step(session)
        print(f"\nJour {i}:")
        print(f"Session: {session}")
        print(f"Fitness: {state['fitness']:.3f}")
        print(f"Fatigue: {state['fatigue']:.3f}")
        print(f"Performance: {state['performance']:.3f}")
        print(f"Forme: {state['form']:.3f}")
        print(f"Récompense: {reward:.2f}")
    
    plot_training_response(env.history)

def plot_training_response(history):
    """Visualise l'évolution des métriques"""
    plt.figure(figsize=(12,8))
    plt.plot(history['fitness'], label='Fitness', marker='o')
    plt.plot(history['fatigue'], label='Fatigue', marker='o')
    plt.plot(history['performance'], label='Performance', marker='o')
    plt.plot(history['form'], label='Forme', marker='o')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Évolution des métriques d\'entraînement')
    plt.xlabel('Jours')
    plt.ylabel('Valeur')
    plt.show()

if __name__ == "__main__":
    test_scenarios()