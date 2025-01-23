import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from training_profile import TrainingProfile

class AdvancedSimulator:
   def __init__(self, profile: Optional[TrainingProfile] = None):
       self.profile = profile or TrainingProfile()
       # Pour suivre les séances de la semaine
       self.current_week_sessions = {
           'intensive': 0,  # Nombre de séances intensives
           'training_days': 0,  # Nombre de jours d'entraînement
       }
       self.week_history = []  # Historique des séances
       self.reset()

   def reset(self):
       """Réinitialise le simulateur"""
       self.fitness = 50
       self.fatigue = 10
       self.form = self.fitness - self.fatigue  # Forme initiale
       self.current_week = 0
       self.week_day = 0
       self.weekly_load = 0  # Charge accumulée cette semaine
       self.target_load = self.profile.charges_hebdo[0]
       self.time_remaining = 84
       self.consecutive_rest = 0
       self.week_history = []
       self.current_week_sessions = {
           'intensive': 0,
           'training_days': 0,
       }
       return self._get_state()

   def get_allowed_actions(self) -> List[Dict]:
       """Retourne les actions autorisées selon les règles physiologiques"""
       allowed_actions = []
       charge_restante = self.target_load - self.weekly_load
       
       # Repos toujours possible sauf si trop de repos consécutifs
       if self.consecutive_rest < 3:  # Max 3 jours de repos consécutifs
           allowed_actions.append({'type': 0, 'charge': 0})
       
       # Compter séances consécutives et intensives cette semaine
       consecutive_training = 0
       intensives_this_week = 0
       for action in self.week_history[-3:]:  # Regarder les 3 derniers jours
           if action['type'] > 0:  # Si entraînement
               consecutive_training += 1
           if action['type'] == 2:  # Si intensif
               intensives_this_week += 1
               
       # Entraînement possible si :
       # - Pas plus de 2 jours consécutifs
       # - Fatigue pas trop haute
       # - Il reste de la charge à faire
       if consecutive_training < 2 and self.fatigue < 55 and charge_restante > 0:
           
           # Endurance possible si forme > 10
           if self.form > 10:
               charge_max = min(charge_restante, self.target_load * 0.25)
               for level in range(5):
                   allowed_actions.append({
                       'type': 1,
                       'charge': (level + 1) * charge_max / 5
                   })
           
           # Intensif possible si :
           # - Pas d'intensif hier
           # - Max 2 par semaine
           # - Forme suffisante
           if (not self.week_history or self.week_history[-1]['type'] != 2) and \
              intensives_this_week < 2 and self.form > 20:
               charge_max = min(charge_restante, self.target_load * 0.4)
               for level in range(5):
                   allowed_actions.append({
                       'type': 2,
                       'charge': (level + 1) * charge_max / 5
                   })
       
       # Si aucune action possible, forcer le repos
       if not allowed_actions:
           allowed_actions.append({'type': 0, 'charge': 0})
           
       return allowed_actions

   def step(self, action: Dict) -> Tuple[np.ndarray, float, bool]:
       # Mise à jour de l'historique des séances
       self.week_history.append(action)
       if len(self.week_history) > 7:
           self.week_history.pop(0)

       # Récupération du type et de la charge de la séance
       session_type = action['type']
       session_charge = action['charge']

       # Mise à jour selon le type de séance
       if session_type == 0:  # Repos
           self.consecutive_rest += 1
           self.fatigue = max(0, self.fatigue - 5)  # Récupération
           
       else:  # Entraînement
           self.consecutive_rest = 0
           self.current_week_sessions['training_days'] += 1
           
           if session_type == 2:  # Intensif
               self.current_week_sessions['intensive'] += 1
               self.fatigue += 10  # Plus de fatigue pour l'intensif
           else:  # Endurance
               self.fatigue += 5
           
           # Mise à jour de la charge hebdomadaire
           self.weekly_load += session_charge
       
       # Mise à jour de la forme
       self.form = max(0, self.fitness - self.fatigue)
       
       # Passage au jour suivant
       self.week_day += 1
       
       # Si fin de semaine
       if self.week_day >= 7:
           self.week_day = 0
           self.current_week += 1
           self.weekly_load = 0
           self.week_history = []  # On vide l'historique en début de semaine
           self.current_week_sessions = {
               'intensive': 0,
               'training_days': 0,
           }
           if self.current_week < len(self.profile.charges_hebdo):
               self.target_load = self.profile.charges_hebdo[self.current_week]
       
       # Mise à jour du temps
       self.time_remaining -= 1
       
       # Calcul récompense et vérification fin
       reward = self._calculate_reward(session_charge)
       done = (self.time_remaining <= 0 or 
               self.fatigue > 80 or 
               self.consecutive_rest > 4)
       
       return self._get_state(), reward, done

   def _calculate_reward(self, session_charge: float) -> float:
       reward = 0
       
       # 1. Récompense quotidienne simple
       if session_charge > 0:  # Si jour d'entraînement
           # Récompense si la charge est raisonnable par rapport à la cible hebdo
           charge_ratio = session_charge / self.target_load
           if 0.1 <= charge_ratio <= 0.3:  # Entre 10% et 30% de la charge hebdo
               reward += 5
           else:
               reward -= 2
       
       # 2. Récompense de fin de semaine
       if self.week_day == 6:
           target_ratio = self.weekly_load / self.target_load
           if 0.9 <= target_ratio <= 1.1:  # ±10% de la cible
               reward += 20
           else:
               reward -= 10 * min(abs(target_ratio - 1.0), 1.0)  # Pénalité limitée
       
       # 3. Petites pénalités pour mauvais états
       reward -= max(0, (self.fatigue - 60) / 10)  # Fatigue excessive
       reward -= max(0, (10 - self.form) / 5)      # Forme trop basse
           
       return float(reward)  # Assurer que c'est un float

   def _get_state(self) -> np.ndarray:
       return np.array([
           self.fitness,
           self.fatigue,
           self.form,
           self.weekly_load,
           self.target_load,
           self.time_remaining
       ], dtype=np.float32)