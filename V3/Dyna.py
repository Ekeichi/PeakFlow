from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict
import random
import json

class TrainingType(Enum):
    """ Type d'entrainement possible par l'environement """

    REPOS = "repos"
    ENDURANCE = "endurance"
    SEUIL = "seuil"
    INTERVAL = "interval"
    COTES = "cotes"
    FARTLEK = "fartlek"
    LONG = "long"
    CROSS_VELO = "cross_velo"
    CROSS_NATATION = "cross_natation"
    FORCE = "force"

class WeatherCondition(Enum):
    """ Conditions météorologiques possible par l'environement """

    IDEAL = "ideal"
    CHAUD = "chaud"
    FROID = "froid"
    PLUIE = "pluie"
    VENT = "vent"

@dataclass
class TrainingZones:
    """Zones de fréquence cardiaque personnalisées"""
    z1: Tuple[int, int]  # Récupération
    z2: Tuple[int, int]  # Endurance fondamentale
    z3: Tuple[int, int]  # Seuil aérobie
    z4: Tuple[int, int]  # Seuil anaérobie
    z5: Tuple[int, int]  # VO2max

    @classmethod
    def calculate_from_fcmax(cls, fc_max: int):
        """ calcul des zones en fct de la fcmax de l'athlète """
        return cls(
            z1=(int(fc_max * 0.5), int(fc_max * 0.6)),
            z2=(int(fc_max * 0.6), int(fc_max * 0.7)),
            z3=(int(fc_max * 0.7), int(fc_max * 0.8)),
            z4=(int(fc_max * 0.8), int(fc_max * 0.9)),
            z5=(int(fc_max * 0.9), int(fc_max))
        )

class MarathonTrainingState:
    def __init__(self):
        # Paramètres du modèle de Bannister
        self.tau_fatigue = 15  # constante de temps fatigue
        self.tau_fitness = 45  # constante de temps fitness
        
        # États du modèle
        self.fitness = 0.0
        self.fatigue = 0.0
        self.performance = 0.0  # Performance = (Fitness - Fatigue)/2
        self.forme = 0.0       # Forme = Fitness - 2*Fatigue
        
        # Autres attributs
        self.fc_repos = 60
        self.vma = 15.0
        self.volume_hebdo = 0.0
        self.derniers_entrainements = []
        self.blessures_actives = []
        self.risque_blessure = 0.0
        self.jours_avant_marathon = 120
        self.meteo = WeatherCondition.IDEAL
        self.temperature = 20.0
        self.zones_fc = TrainingZones.calculate_from_fcmax(194)

    def update_bannister(self, effort: float):
        """Met à jour le modèle de Bannister après un entraînement"""
        # Mise à jour fatigue et fitness selon vos formules
        self.fatigue = effort + np.exp(-1/self.tau_fatigue) * self.fatigue
        self.fitness = effort + np.exp(-1/self.tau_fitness) * self.fitness
        
        # Calcul des indicateurs
        self.performance = (self.fitness - self.fatigue) / 2
        self.forme = self.fitness - 2 * self.fatigue

    def discretize(self) -> tuple:
        """Discrétise l'état pour le Q-learning"""
        return (
            round(self.fitness * 10),
            round(self.fatigue * 10),
            round(self.performance * 10),
            round(self.vma * 2),
            round(self.volume_hebdo / 10),
            round(self.risque_blessure * 10),
            min(120, self.jours_avant_marathon),
            round(self.temperature / 5)
        )

class TrainingAction:
    def __init__(self, 
                 type: TrainingType,
                 duree: int,  # minutes
                 intensite: float,  # 0-1
                 zone_fc: int):  # 1-5
        self.type = type
        self.duree = duree
        self.intensite = intensite
        self.zone_fc = zone_fc
    
    def discretize(self) -> tuple:
        return (
            self.type.value,
            self.duree // 15,  # tranches de 15 minutes
            round(self.intensite * 5),
            self.zone_fc
        )

class MarathonEnvironment:
    def __init__(self):
        self.state = MarathonTrainingState()
        self.history = []
        
        # Définition des zones appropriées par type d'entraînement
        self.zones_par_type = {
            TrainingType.REPOS: [1],
            TrainingType.ENDURANCE: [2, 3],
            TrainingType.SEUIL: [4],
            TrainingType.INTERVAL: [4, 5],
            TrainingType.COTES: [4, 5],
            TrainingType.FARTLEK: [3, 4],
            TrainingType.LONG: [2],
            TrainingType.CROSS_VELO: [1, 2, 3],
            TrainingType.CROSS_NATATION: [1, 2, 3],
            TrainingType.FORCE: [1, 2]
        }
        
        # Définition des durées appropriées par type d'entraînement
        self.durees_par_type = {
            TrainingType.REPOS: [0],
            TrainingType.ENDURANCE: [45, 60],
            TrainingType.SEUIL: [30, 45],
            TrainingType.INTERVAL: [30, 45],
            TrainingType.COTES: [30, 45],
            TrainingType.FARTLEK: [30, 45],
            TrainingType.LONG: [90, 120],
            TrainingType.CROSS_VELO: [30, 45, 60],
            TrainingType.CROSS_NATATION: [30, 45],
            TrainingType.FORCE: [30, 45]
        }
    
    def reset(self):
        self.state = MarathonTrainingState()
        self.history = []
        return self.state
    
    def step(self, action: TrainingAction) -> Tuple[MarathonTrainingState, float, bool]:
        # Sauvegarder l'historique
        self.history.append((self.state, action))
        
        # Copier l'état actuel
        new_state = MarathonTrainingState()
        new_state.__dict__.update(self.state.__dict__)
        
        # Mettre à jour l'historique des entraînements
        new_state.derniers_entrainements.append(action.type)
        if len(new_state.derniers_entrainements) > 7:
            new_state.derniers_entrainements.pop(0)
        
        # Calculer la charge d'entraînement et appliquer le modèle de Bannister
        training_load = self._calculate_training_load(action)
        new_state.update_bannister(training_load)
        
        # Calculer la récompense
        reward = self._calculate_reward(new_state, action, training_load)
        
        # Mise à jour du temps restant
        new_state.jours_avant_marathon = max(0, new_state.jours_avant_marathon - 1)
        
        # Vérifier si l'entraînement est terminé
        done = new_state.jours_avant_marathon <= 0
        
        self.state = new_state
        return new_state, reward, done

    def _calculate_training_load(self, action: TrainingAction) -> float:
        """Calcule la charge d'entraînement normalisée"""
        if action.type == TrainingType.REPOS:
            return 0.0
        
        # Calcul de l'effort avec les zone_weights exponentiels
        zone_weights = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        effort = (action.duree/60) * np.exp(zone_weights[action.zone_fc])
        
        # Ajustements selon le type d'entraînement
        type_factors = {
            TrainingType.INTERVAL: 1.1,  # Réduit de 1.2 à 1.1
            TrainingType.COTES: 1.1,    # Réduit de 1.2 à 1.1
            TrainingType.LONG: 1.3,     # Augmenté de 1.1 à 1.3
            TrainingType.ENDURANCE: 1.2, # Nouveau facteur
            TrainingType.CROSS_VELO: 0.8,
            TrainingType.CROSS_NATATION: 0.7,
            TrainingType.FORCE: 0.6
        }
        if action.type in type_factors:
            effort *= type_factors[action.type]
        
        # Normalisation
        # Une séance maximale serait : 120 minutes en zone 5 (exp(5) ≈ 148)
        max_possible_effort = (120/60) * np.exp(5) * 1.2  # 1.2 est le facteur max
        
        # Normaliser entre 0 et 1
        normalized_effort = effort / max_possible_effort
        
        return normalized_effort

    def _calculate_reward(self, state: MarathonTrainingState, action: TrainingAction, training_load: float) -> float:
        reward = 0
        
        # Progression de performance
        performance_delta = state.performance - (state.fitness - state.fatigue)/2
        reward += performance_delta * 10
        
        # Vérifier si une sortie longue a déjà été faite dans les 7 derniers jours
        has_long_run_this_week = False
        if len(state.derniers_entrainements) > 0:
            last_seven_days = state.derniers_entrainements[-7:]
            has_long_run_this_week = TrainingType.LONG in last_seven_days
        
        # Gestion des sorties longues
        if action.type == TrainingType.LONG and has_long_run_this_week:
            reward -= 5.0  # Pénalité réduite
        elif action.type == TrainingType.LONG and state.jours_avant_marathon > 60:
            reward += 4.0  # Bonus augmenté pour début de préparation
        
        # Bonus pour séquences d'entraînement optimales
        if len(state.derniers_entrainements) >= 3:
            last_three = state.derniers_entrainements[-3:]
            # Séquences positives
            good_sequences = [
                [TrainingType.ENDURANCE, TrainingType.INTERVAL, TrainingType.REPOS],
                [TrainingType.LONG, TrainingType.REPOS, TrainingType.INTERVAL],
                [TrainingType.INTERVAL, TrainingType.REPOS, TrainingType.ENDURANCE]
            ]
            if last_three in good_sequences:
                reward += 2.0
        
        # Bonus pour la distribution des types d'entraînement
        if len(state.derniers_entrainements) >= 7:
            type_counts = {}
            for t in state.derniers_entrainements[-7:]:
                type_counts[t] = type_counts.get(t, 0) + 1
            
            # Pénaliser la sur-utilisation d'un type
            for count in type_counts.values():
                if count > 2:
                    reward -= 1.0
        
        # Bonus pour respect des zones et durées
        if action.zone_fc in self.zones_par_type[action.type]:
            reward += 2.0
        if action.duree in self.durees_par_type[action.type]:
            reward += 1.0
        
        # Pénalité pour surcharge
        if state.fatigue > 1.5 * state.fitness:
            reward -= 5.0
            
        return reward
    

class AdvancedDynaQMarathon:
    def __init__(self, 
                 n_planning_steps: int = 10,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        self.Q = defaultdict(lambda: defaultdict(float))
        self.model = {}
        self.n_planning_steps = n_planning_steps
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Générer l'espace d'actions
        self.actions = self._generate_action_space()
        
        # Historique d'apprentissage
        self.training_history = []
        self.rewards_history = []
    
    def _generate_action_space(self) -> List[TrainingAction]:
        """Génère l'espace d'actions discrétisé"""
        actions = []
        
        # Action de repos
        actions.append(TrainingAction(TrainingType.REPOS, 0, 0, 1))
        
        # Autres types d'entraînement
        for training_type in [t for t in TrainingType if t != TrainingType.REPOS]:
            for duree in [30, 45, 60, 90, 120]:
                for intensite in [0.6, 0.7, 0.8, 0.9]:
                    for zone in range(1, 6):
                        if self._is_valid_combination(training_type, duree, intensite, zone):
                            actions.append(TrainingAction(training_type, duree, intensite, zone))
        
        return actions
    
    def _is_valid_combination(self, type: TrainingType, duree: int, intensite: float, zone: int) -> bool:
        contraintes_type = {
            TrainingType.REPOS: lambda d, i, z: d == 0 and z == 1,
            TrainingType.ENDURANCE: lambda d, i, z: d in [45, 60, 90] and z in [2, 3],  # Ajout de 90 min
            TrainingType.SEUIL: lambda d, i, z: d in [30, 45] and z == 4,
            TrainingType.INTERVAL: lambda d, i, z: d in [30, 45] and z in [4, 5],
            TrainingType.COTES: lambda d, i, z: d in [30, 45] and z in [4, 5],
            TrainingType.FARTLEK: lambda d, i, z: d in [30, 45] and z in [3, 4],
            TrainingType.LONG: lambda d, i, z: d in [90, 120] and z == 2,
            TrainingType.CROSS_VELO: lambda d, i, z: d in [30, 45, 60] and z in [2, 3],
            TrainingType.CROSS_NATATION: lambda d, i, z: d in [30, 45] and z in [2, 3],
            TrainingType.FORCE: lambda d, i, z: d in [30, 45] and z in [1, 2]
        }
        
        if type in contraintes_type:
            return contraintes_type[type](duree, intensite, zone)
        return False
    
    def get_action(self, state: MarathonTrainingState) -> TrainingAction:
        """Sélectionne une action selon la politique epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
            
        state_key = state.discretize()
        if state_key not in self.Q:
            return random.choice(self.actions)
            
        return max(self.actions, 
                  key=lambda a: self.Q[state_key][a.discretize()])
    
    def learn(self, 
              state: MarathonTrainingState,
              action: TrainingAction,
              reward: float,
              next_state: MarathonTrainingState):
        """Met à jour la fonction Q et le modèle, puis effectue la planification"""
        state_key = state.discretize()
        action_key = action.discretize()
        next_state_key = next_state.discretize()
        
        # Mise à jour Q-learning directe
        best_next_value = max([self.Q[next_state_key][a.discretize()] 
                             for a in self.actions])
        
        td_target = reward + self.gamma * best_next_value
        current_q = self.Q[state_key][action_key]
        self.Q[state_key][action_key] = current_q + self.lr * (td_target - current_q)
        
        # Mise à jour du modèle
        self.model[(state_key, action_key)] = (reward, next_state_key)
        
        # Planification
        self.plan()
        
        # Sauvegarder l'historique
        self.training_history.append((state_key, action_key, reward, next_state_key))
        self.rewards_history.append(reward)
    
    def plan(self):
        """Effectue n_planning_steps mises à jour en utilisant le modèle"""
        if not self.model:
            return
            
        for _ in range(self.n_planning_steps):
            # Choisir une expérience aléatoire du modèle
            state_action = random.choice(list(self.model.keys()))
            reward, next_state_key = self.model[state_action]
            state_key, action_key = state_action
            
            # Mise à jour Q basée sur le modèle
            best_next_value = max([self.Q[next_state_key][a.discretize()] 
                                 for a in self.actions])
            td_target = reward + self.gamma * best_next_value
            current_q = self.Q[state_key][action_key]
            self.Q[state_key][action_key] = current_q + self.lr * (td_target - current_q)

    def get_training_recommendation(self, state: MarathonTrainingState) -> Dict:
        """Génère une recommandation d'entraînement détaillée"""
        action = self.get_action(state)
        
        training_descriptions = {
            TrainingType.REPOS: "Journée de repos pour la récupération",
            TrainingType.ENDURANCE: "Entraînement d'endurance à allure modérée",
            TrainingType.SEUIL: "Entraînement au seuil anaérobie",
            TrainingType.INTERVAL: "Séance d'intervalles haute intensité",
            TrainingType.COTES: "Entraînement en côtes",
            TrainingType.FARTLEK: "Fartlek - jeu de vitesse",
            TrainingType.LONG: "Sortie longue",
            TrainingType.CROSS_VELO: "Cross-training vélo",
            TrainingType.CROSS_NATATION: "Cross-training natation",
            TrainingType.FORCE: "Renforcement musculaire"
        }
        
        zone_descriptions = {
            1: "Zone 1 (Récupération active)",
            2: "Zone 2 (Endurance fondamentale)",
            3: "Zone 3 (Seuil aérobie)",
            4: "Zone 4 (Seuil anaérobie)",
            5: "Zone 5 (VO2max)"
        }
        
        return {
            "type": action.type.value,
            "description": training_descriptions[action.type],
            "duree": action.duree,
            "intensite": action.intensite,
            "zone_fc": zone_descriptions[action.zone_fc],
            "fc_cible": state.zones_fc.__dict__[f'z{action.zone_fc}'],
            "confiance": self.Q[state.discretize()][action.discretize()]
        }
    
    def save_model(self, filepath: str):
        model_data = {
            'Q': {str(state): {str(action): value 
                for action, value in actions.items()}
                for state, actions in self.Q.items()},
            'model': {str(state_action): (reward, str(next_state))
                    for state_action, (reward, next_state) in self.model.items()},
            'params': {
                'n_planning_steps': self.n_planning_steps,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)

    def load_model(self, filepath: str):
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.Q = defaultdict(lambda: defaultdict(float))
        for state_str, actions in model_data['Q'].items():
            state = eval(state_str)
            for action_str, value in actions.items():
                action = eval(action_str)
                self.Q[state][action] = value
        
        self.model = {eval(k): (v[0], eval(v[1])) 
                    for k, v in model_data['model'].items()}
        
        for param, value in model_data['params'].items():
            setattr(self, param, value)



def train_agent(episodes: int = 5000):
    """Fonction pour entraîner l'agent"""
    env = MarathonEnvironment()
    agent = AdvancedDynaQMarathon()
    
    # Pour le epsilon décroissant
    initial_epsilon = 0.9
    final_epsilon = 0.1
    epsilon_decay = (initial_epsilon - final_epsilon) / episodes
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Mettre à jour epsilon
        agent.epsilon = initial_epsilon - episode * epsilon_decay
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    
    return agent, env

if __name__ == "__main__":
    # Entraîner l'agent
    trained_agent, env = train_agent(episodes=5000)

    # Sauvegarder le modèle
    trained_agent.save_model('trained_marathon_model.json')
    
    # Test de l'agent entraînés
    state = env.reset()
    print("\nTest de l'agent sur 10 jours :")
    for day in range(10):
        recommendation = trained_agent.get_training_recommendation(state)
        
        print(f"\nJour {day + 1}:")
        print(f"État actuel:")
        print(f"- Fitness: {state.fitness:.2f}")
        print(f"- Fatigue: {state.fatigue:.2f}")
        print(f"- Performance: {state.performance:.2f}")
        
        print(f"\nRecommandation:")
        print(f"- Type: {recommendation['description']}")
        print(f"- Durée: {recommendation['duree']} minutes")
        print(f"- Zone FC: {recommendation['zone_fc']}")
        print(f"- FC cible: {recommendation['fc_cible']}")
        
        action = trained_agent.get_action(state)
        next_state, reward, _ = env.step(action)
        print(f"Récompense: {reward:.2f}")
        
        state = next_state