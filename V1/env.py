import gymnasium as gym
import numpy as np

class TrainingEnv(gym.Env):
    def __init__(self, simulateur):
        super(TrainingEnv, self).__init__()

        self.simulateur = simulateur

        # Espace d'observation
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),  
            high=np.array([100, 100, 100, 1000, 1000, 84]),  
            dtype=np.float32
        )

        # Action space: [type_séance (0-2), niveau_charge (0-4)]
        self.action_space = gym.spaces.MultiDiscrete([3, 5])

    def reset(self, *, seed=None, options=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.state = self.simulateur.reset()
        return self.state, {}

    def step(self, action):
        # Convertir l'action MultiDiscrete en dictionnaire pour le simulateur
        action_dict = {
            'type': int(action[0]),
            'charge': self._convert_charge_level(action[1], int(action[0]))  # On passe le type de séance
        }
        
        next_state, reward, done = self.simulateur.step(action_dict)
        
        terminated = done
        truncated = False
        
        return next_state, reward, terminated, truncated, {}

    def _convert_charge_level(self, level: int, session_type: int) -> float:
        """
        Convertit le niveau de charge (0-4) en valeur réelle
        selon le type de séance
        """
        if session_type == 0:  # Repos
            return 0
        
        # Base de calcul différente selon le type
        if session_type == 1:  # Endurance
            charge_max = self.simulateur.target_load * 0.25  # 25% max
        else:  # Intensif
            charge_max = self.simulateur.target_load * 0.4   # 40% max
            
        return (level + 1) * charge_max / 5

    def render(self, mode="human"):
        print(f"État actuel : {self.state}")