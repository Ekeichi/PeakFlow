import numpy as np
from dataclasses import dataclass

@dataclass
class TrainingProfile:
    """Profil de charge sur 12 semaines"""
    def __init__(self):
        self.semaines_totales = 12
        self.volume_initial = 30
        self.intensite_moyenne = 6
        self.progression = 0.12
        self.tapering_start = 10
        self.charges_hebdo = self._calculate_weekly_loads()
        
    def _calculate_weekly_loads(self):
        charges = []
        volume_actuel = self.volume_initial
        
        for semaine in range(1, self.semaines_totales + 1):
            if semaine >= self.tapering_start:
                volume_actuel *= 0.85
            elif semaine % 4 == 0:  # Semaine de récupération 1
                volume_actuel *= 0.85
            elif semaine % 8 == 0:  # Semaine de récupération 2
                volume_actuel *= 0.92
            else:
                volume_actuel *= (1 + self.progression)
                
            charge = volume_actuel * self.intensite_moyenne
            charges.append(charge)
            
        return charges