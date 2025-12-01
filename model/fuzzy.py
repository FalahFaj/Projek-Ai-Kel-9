import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

def build_fuzzy_system():

    # 1. Antecedents (Input)
    biaya = ctrl.Antecedent(np.arange(0, 50001, 1000), 'biaya')
    jarak = ctrl.Antecedent(np.arange(0, 5001, 100), 'jarak')
    rasa  = ctrl.Antecedent(np.arange(1, 6, 0.1), 'rasa')
    
    # 2. Consequent (Output)
    skor = ctrl.Consequent(np.arange(0, 101, 1), 'skor')

    # 3. Membership Functions
    # Biaya
    biaya['murah']  = fuzz.trapmf(biaya.universe, [0, 0, 12000, 15000])
    biaya['sedang'] = fuzz.trimf(biaya.universe, [12000, 18000, 25000])
    biaya['mahal']  = fuzz.trapmf(biaya.universe, [20000, 30000, 50000, 50000])

    # Jarak
    jarak['dekat']  = fuzz.trapmf(jarak.universe, [0, 0, 800, 1200])
    jarak['sedang'] = fuzz.trimf(jarak.universe, [800, 1800, 3000])
    jarak['jauh']   = fuzz.trapmf(jarak.universe, [2000, 3500, 5000, 5000])

    # Rasa
    rasa['biasa']   = fuzz.trapmf(rasa.universe, [1, 1, 3.5, 4.2])
    rasa['enak']    = fuzz.trapmf(rasa.universe, [3.8, 4.5, 5, 5])

    # Output Skor
    skor['rendah']  = fuzz.trimf(skor.universe, [0, 0, 50])
    skor['sedang']  = fuzz.trimf(skor.universe, [40, 60, 80])
    skor['tinggi']  = fuzz.trimf(skor.universe, [70, 100, 100])

    # 4. Rules
    rule1 = ctrl.Rule(biaya['murah'] & rasa['enak'], skor['tinggi'])
    rule2 = ctrl.Rule(biaya['murah'] & jarak['dekat'], skor['tinggi'])
    rule3 = ctrl.Rule(biaya['mahal'] & rasa['enak'], skor['sedang'])
    rule4 = ctrl.Rule(biaya['mahal'] & rasa['biasa'], skor['rendah'])
    rule5 = ctrl.Rule(jarak['jauh'] & rasa['biasa'], skor['rendah'])
    rule6 = ctrl.Rule(biaya['sedang'] & rasa['enak'] & jarak['dekat'], skor['tinggi'])

    rekomendasi_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    simulation = ctrl.ControlSystemSimulation(rekomendasi_ctrl)
    
    return simulation

def calculate_complex_score(row, simulation):
    """Menghitung skor fuzzy untuk satu baris data"""
    try:
        simulation.input['biaya'] = row['Biaya']
        simulation.input['jarak'] = row['Jarak_Meter']
        simulation.input['rasa']  = row['Rating_Rasa']
        simulation.compute()
        return simulation.output['skor']
    except Exception:
        return 50.0

def apply_fuzzy_scoring(df):
    print("--- [Fuzzy Logic] Menghitung Skor Rekomendasi... ---")
    sim = build_fuzzy_system()
    
    # Cleaning data numerik sementara untuk perhitungan fuzzy
    temp_df = df.copy()
    cols = ['Biaya', 'Jarak_Meter', 'Rating_Rasa']
    for col in cols:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
        
    # Apply perhitungan
    df['Fuzzy_Score'] = temp_df.apply(lambda row: calculate_complex_score(row, sim), axis=1)
    return df