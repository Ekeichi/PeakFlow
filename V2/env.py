import numpy as np

class MarathonEnv:
    def __init__(self):
        # État initial
        self.state = {
            'fitness': 0.2,         # Niveau de forme (0-1)
            'fatigue': 0.0,         # Niveau de fatigue (0-1)
            'days_to_goal': 90,     # Jours avant le marathon
            'last_week_volume': 20,  # Volume de la semaine précédente en km
            'performance': 0.1,  # Nouvelle métrique
            'form': 0.0         # Nouvelle métrique
        }

        # Constantes du modèle
        self.tau_fatigue = 15   # Décroissance de la fatigue
        self.tau_fitness = 45   # Décroissance de la fitness

    def step(self, action):
        # Vérifier la sécurité de l'action
        if not self._is_safe(action):
            print("⚠️ Action non sécurisée")
            return self.state, -10, True, {"error": "unsafe_action"}
            
        # Simuler les effets
        new_state = self._simulate_effects(action)
        
        # Calculer la récompense
        reward = self._calculate_reward(new_state)
        
        # Vérifier si terminé
        done = new_state['days_to_goal'] <= 0
        
        self.state = new_state
        return new_state, reward, done, {}
    
    def _is_safe(self, action):
        # Si repos, toujours autorisé
        if action['type'] == 'rest':
            return True
            
        # Utilisation de la forme comme indicateur principal
        if self.state['form'] < -0.3:  # forme négative significative
            if action['intensity'] > 0.7:
                print("⚠️ Forme trop basse pour séance intense")
                return False
            if action['volume'] > 10:
                print("⚠️ Forme trop basse pour long volume")
                return False
                
        # Forme très négative : uniquement sessions faciles
        if self.state['form'] < -0.5:
            if action['intensity'] > 0.6:
                print("⚠️ Forme très basse - uniquement sessions faciles permises")
                return False
                
        # Protection volume
        if action['volume'] > self.state['last_week_volume'] * 1.2:
            print("⚠️ Augmentation volume trop importante")
            return False
                
        return True
            
    
    def _simulate_effects(self, action):
        new_state = self.state.copy()
        
        # Normalisation de l'effort
        effort = (action['volume'] * action['intensity']) / 100  # Division pour réduire l'échelle
        
        if action['type'] == 'rest':
            new_state['fatigue'] = self.state['fatigue'] * np.exp(-1/self.tau_fatigue)
            new_state['fitness'] = self.state['fitness'] * np.exp(-1/self.tau_fitness)
        else:
            new_state['fatigue'] = (effort + np.exp(-1/self.tau_fatigue) * self.state['fatigue'])
            new_state['fitness'] = (effort + np.exp(-1/self.tau_fitness) * self.state['fitness'])
        
        # Calcul des métriques dérivées
        new_state['performance'] = (new_state['fitness'] - new_state['fatigue'])/2
        if new_state['performance'] < 0.05:  # Seuil minimal plus bas
            new_state['performance'] = 0.05
        new_state['form'] = new_state['fitness'] - 2*new_state['fatigue']
        
        # Borner les valeurs
        new_state['fatigue'] = min(1.0, max(0.0, new_state['fatigue']))
        new_state['fitness'] = min(1.0, max(0.0, new_state['fitness']))
        
        new_state['days_to_goal'] -= 1
        return new_state
    
    def _calculate_reward(self, new_state):
        reward = 0

        # Bonus pour maintenir un ratio sain
        ratio = (new_state['fatigue'] / new_state['performance']) * 100
        if ratio < 150:
            reward += 5
        elif ratio < 200:
            reward += 2
        
        # Récompense basée sur l'amélioration de performance
        perf_delta = new_state['performance'] - self.state['performance']
        reward += 50 * perf_delta
        
        # Récompense/pénalité basée sur la forme
        if new_state['form'] > 0.1:
            reward += 2  # Bonne forme
        elif new_state['form'] < -0.3:
            reward -= 5  # Forme trop négative
            
        # Pénalité pour fatigue excessive
        if new_state['fatigue'] > 0.3:
            reward -= (new_state['fatigue'] - 0.3) * 10
            
        # Bonus pour progression fitness
        fitness_gain = new_state['fitness'] - self.state['fitness']
        if fitness_gain > 0:
            reward += 20 * fitness_gain
            
        return reward

# Test de l'environnement
env = MarathonEnv()
print("État initial:", env.state)

# # Test avec différentes séances
# week_plan = [
#     {'type': 'easy', 'volume': 10, 'intensity': 0.6},
#     {'type': 'rest', 'volume': 0, 'intensity': 0},
#     {'type': 'tempo', 'volume': 12, 'intensity': 0.8},
#     {'type': 'easy', 'volume': 8, 'intensity': 0.6},
#     {'type': 'long_run', 'volume': 20, 'intensity': 0.7}
# ]

# for i, session in enumerate(week_plan, 1):
#     print(f"\nJour {i} - Session: {session}")
#     state, reward, done, info = env.step(session)
#     print(f"Nouvel état: {state}")
#     print(f"Récompense: {reward}")
#     print(f"Terminé: {done}")