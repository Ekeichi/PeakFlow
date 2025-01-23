from training_profile import TrainingProfile
from simulateur import AdvancedSimulator
from env import TrainingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Création du profil et du simulateur
profile = TrainingProfile()
simulateur = AdvancedSimulator(profile)

# Initialisation de l'environnement vectorisé
vec_env = DummyVecEnv([lambda: TrainingEnv(simulateur)])

# Initialisation et entraînement du modèle
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=500000)

# Test du modèle entraîné
print("\nGénération du programme d'entraînement sur 84 jours :")
print("-" * 50)

state = vec_env.reset()
program = []

def session_type_to_string(session_type):
    return {
        0: "Repos",
        1: "Endurance",
        2: "Intensif"
    }[session_type]

def convert_charge_level(level, target_load, session_type):
    if session_type == 0:  # Repos
        return 0
    if session_type == 1:  # Endurance
        charge_max = target_load * 0.25
    else:  # Intensif
        charge_max = target_load * 0.4
    return (level + 1) * charge_max / 5

for day in range(84):
    action, _ = model.predict(state)
    state, reward, done, _ = vec_env.step(action)
    
    # Extraire les métriques de l'état
    fitness = state[0][0]
    fatigue = state[0][1]
    form = state[0][2]
    weekly_load = state[0][3]
    target_load = state[0][4]
    
    # Convertir l'action en valeurs lisibles
    session_type = int(action[0][0])  # Ajout de [0] pour accéder à la première dimension
    charge = convert_charge_level(action[0][1], target_load, session_type)
    
    print(f"Jour {day + 1:3d} : {session_type_to_string(session_type):8s} | "
          f"Charge: {charge:4.1f} | "
          f"Forme: {form:6.1f} | Fatigue: {fatigue:6.1f} | "
          f"Charge hebdo: {weekly_load:6.1f} | Cible: {target_load:6.1f}")
    
    if done:
        print("Simulation terminée.")
        break

# Sauvegarder le modèle
model.save("training_model_v1")