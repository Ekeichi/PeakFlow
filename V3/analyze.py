from Dyna import MarathonEnvironment, AdvancedDynaQMarathon
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_full_training_plan():
    # Charger le modèle entraîné
    agent = AdvancedDynaQMarathon()
    agent.load_model('trained_marathon_model.json')
    
    # Générer le plan
    env = MarathonEnvironment()
    state = env.reset()
    training_data = []
    
    for day in range(120):
        action = agent.get_action(state)
        next_state, reward, _ = env.step(action)
        
        # Enregistrer les données
        training_data.append({
            'jour': day + 1,
            'type': action.type.value,
            'duree': action.duree,
            'zone_fc': action.zone_fc,
            'fitness': next_state.fitness,
            'fatigue': next_state.fatigue,
            'performance': next_state.performance
        })
        
        state = next_state
    
    return training_data

def plot_physiological_values(training_data):
    df = pd.DataFrame(training_data)
    
    # Configuration du style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Tracer les courbes
    ax.plot(df['jour'], df['fitness'], label='Fitness', linewidth=2)
    ax.plot(df['jour'], df['fatigue'], label='Fatigue', linewidth=2)
    ax.plot(df['jour'], df['performance'], label='Performance', linewidth=2)
    
    # Personnalisation du graphique
    ax.set_title('Évolution des paramètres physiologiques sur 120 jours', fontsize=14, pad=20)
    ax.set_xlabel('Jours', fontsize=12)
    ax.set_ylabel('Valeur', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Ajouter des phases d'entraînement
    ax.axvspan(0, 48, alpha=0.2, color='green', label='Phase de base')
    ax.axvspan(48, 96, alpha=0.2, color='yellow', label='Phase spécifique')
    ax.axvspan(96, 120, alpha=0.2, color='red', label='Affûtage')
    
    plt.tight_layout()
    plt.savefig('evolution_physiologique.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_training_distribution(training_data):
    df = pd.DataFrame(training_data)
    
    # Analyser par phase
    phases = {
        'Base (1-48j)': df[df['jour'] <= 48],
        'Spécifique (49-96j)': df[(df['jour'] > 48) & (df['jour'] <= 96)],
        'Affûtage (97-120j)': df[df['jour'] > 96]
    }
    
    distribution = {}
    for phase_name, phase_data in phases.items():
        type_counts = phase_data['type'].value_counts()
        distribution[phase_name] = type_counts
    
    return pd.DataFrame(distribution)

if __name__ == "__main__":
    # Générer le plan
    print("Génération du plan d'entraînement...")
    training_data = generate_full_training_plan()
    
    # Sauvegarder en CSV
    print("Sauvegarde du plan en CSV...")
    df = pd.DataFrame(training_data)
    df.to_csv('plan_marathon.csv', index=False)
    
    # Créer le graphique
    print("Création du graphique d'évolution physiologique...")
    plot_physiological_values(training_data)
    
    # Analyser la distribution des entraînements
    print("\nAnalyse de la distribution des types d'entraînement par phase :")
    distribution = analyze_training_distribution(training_data)
    print(distribution)
    
    print("\nPlan d'entraînement généré et analyses terminées!")
    print("Fichiers créés :")
    print("- plan_marathon.csv : Plan détaillé")
    print("- evolution_physiologique.png : Graphique des paramètres")