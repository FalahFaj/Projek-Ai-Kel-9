# ==========================================
# FILE: model/fuzzy.py
# ==========================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- PENGATURAN PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_folder = os.path.join(project_root, 'Data')

input_file = os.path.join(data_folder, 'data_berlabel.csv')
output_final = os.path.join(data_folder, 'hasil_rekomendasi.csv')

# 1. LOAD DATA
if not os.path.exists(input_file):
    print(f"ERROR: File {input_file} tidak ditemukan. Jalankan K-Means dulu!")
    exit(1)

df = pd.read_csv(input_file)

# 2. LOGIKA FUZZY
biaya = ctrl.Antecedent(np.arange(0, 60001, 1000), 'biaya')
jarak = ctrl.Antecedent(np.arange(0, 12001, 100), 'jarak')
rasa  = ctrl.Antecedent(np.arange(1, 6, 0.1), 'rasa')
rekomendasi = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')

# Membership Functions
biaya['murah']  = fuzz.trimf(biaya.universe, [0, 0, 15000])
biaya['sedang'] = fuzz.trimf(biaya.universe, [10000, 20000, 35000])
biaya['mahal']  = fuzz.trapmf(biaya.universe, [25000, 40000, 60000, 60000])

jarak['dekat']  = fuzz.trimf(jarak.universe, [0, 0, 1500])
jarak['sedang'] = fuzz.trimf(jarak.universe, [1000, 3000, 5000])
jarak['jauh']   = fuzz.trapmf(jarak.universe, [4000, 8000, 12000, 12000])

rasa['kurang'] = fuzz.trimf(rasa.universe, [1, 1, 3])
rasa['biasa']  = fuzz.trimf(rasa.universe, [2, 3, 4])
rasa['enak']   = fuzz.trapmf(rasa.universe, [3.5, 5, 5, 5])

rekomendasi['tidak']  = fuzz.trimf(rekomendasi.universe, [0, 0, 50])
rekomendasi['sedang'] = fuzz.trimf(rekomendasi.universe, [40, 60, 80])
rekomendasi['ya']     = fuzz.trimf(rekomendasi.universe, [70, 100, 100])

# Rules
rule1 = ctrl.Rule(biaya['murah'] & rasa['enak'], rekomendasi['ya'])
rule2 = ctrl.Rule(biaya['sedang'] & rasa['enak'] & jarak['dekat'], rekomendasi['ya'])
rule3 = ctrl.Rule(biaya['mahal'] & rasa['enak'] & jarak['dekat'], rekomendasi['sedang'])
rule4 = ctrl.Rule(jarak['jauh'] & rasa['enak'], rekomendasi['sedang'])
rule5 = ctrl.Rule(rasa['kurang'], rekomendasi['tidak'])
rule6 = ctrl.Rule(biaya['mahal'] & rasa['biasa'], rekomendasi['tidak'])

rekomen_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
rekomen_sim = ctrl.ControlSystemSimulation(rekomen_ctrl)

# 3. HITUNG SKOR
def hitung_skor(row):
    try:
        rekomen_sim.input['biaya'] = row['Biaya']
        rekomen_sim.input['jarak'] = row['Jarak_Meter']
        rekomen_sim.input['rasa']  = row['Rating_Rasa']
        rekomen_sim.compute()
        return rekomen_sim.output['rekomendasi']
    except:
        return 0

print("[INFO] Menghitung skor fuzzy...")
df['Skor_Rekomendasi'] = df.apply(hitung_skor, axis=1)

# 4. VISUALISASI (LANGSUNG SIMPAN PNG)
plt.figure(figsize=(8, 5))
sns.histplot(df['Skor_Rekomendasi'], bins=15, kde=True, color='green')
plt.title('Distribusi Skor Rekomendasi')
plt.xlabel('Skor')
plt.ylabel('Jumlah')
plot_path = os.path.join(data_folder, 'distribusi_skor.png')
plt.savefig(plot_path)
plt.close() # Menutup plot
print(f"[INFO] Grafik distribusi disimpan ke: {plot_path}")

# Simpan Hasil
df.to_csv(output_final, index=False)
print(f"[INFO] Hasil lengkap disimpan di: {output_final}")


# ==========================================
# 5. FITUR AI ASSISTANT (INTERAKTIF)
# ==========================================
def run_ai_assistant():
    print("\n" + "="*50)
    print("      🤖 AI ASSISTANT REKOMENDASI KULINER      ")
    print("==================================================")
    print("Ketik pertanyaanmu, contoh:")
    print(" - 'Saya ingin makanan termurah'")
    print(" - 'Cari tempat makan yang paling dekat'")
    print(" - 'Rekomendasi makanan enak'")
    print(" - 'Tampilkan top 5 rekomendasi'")
    print("(Ketik 'exit' atau 'keluar' untuk berhenti)")
    
    while True:
        user_input = input("\nUser ➤ ").lower()
        
        if user_input in ['exit', 'keluar', 'quit']:
            print("AI   ➤ Terima kasih! Selamat makan 👋")
            break
        
        # Logika Sederhana AI (Keyword Matching)
        hasil = df.copy()
        pesan = ""
        
        # Keyword matching untuk menangkap maksud user
        if 'murah' in user_input or 'terjangkau' in user_input:
            hasil = hasil.sort_values(by='Biaya', ascending=True)
            pesan = "Berikut tempat makan dengan HARGA TERMURAH:"
        elif 'mahal' in user_input or 'sultan' in user_input:
            hasil = hasil.sort_values(by='Biaya', ascending=False)
            pesan = "Berikut tempat makan dengan HARGA TERTINGGI:"
        elif 'dekat' in user_input:
            hasil = hasil.sort_values(by='Jarak_Meter', ascending=True)
            pesan = "Berikut tempat makan TERDEKAT dari kampus:"
        elif 'jauh' in user_input:
            hasil = hasil.sort_values(by='Jarak_Meter', ascending=False)
            pesan = "Berikut tempat makan terjauh:"
        elif 'enak' in user_input or 'lezat' in user_input or 'mantap' in user_input:
            hasil = hasil.sort_values(by='Rating_Rasa', ascending=False)
            pesan = "Berikut tempat makan dengan RASA PALING ENAK:"
        elif 'nyaman' in user_input:
            hasil = hasil.sort_values(by='Rating_Nyaman', ascending=False)
            pesan = "Berikut tempat makan PALING NYAMAN buat nongkrong:"
        elif 'wifi' in user_input:
            hasil = hasil[hasil['Wifi_Score'] >= 1].sort_values(by='Wifi_Score', ascending=False)
            pesan = "Berikut tempat makan dengan fasilitas WIFI:"
        elif 'colokan' in user_input or 'listrik' in user_input:
            hasil = hasil[hasil['Colokan_Score'] >= 1].sort_values(by='Colokan_Score', ascending=False)
            pesan = "Berikut tempat makan dengan fasilitas COLOKAN LISTRIK:"
        else:
            # Default: Berdasarkan Skor Fuzzy Terbaik
            hasil = hasil.sort_values(by='Skor_Rekomendasi', ascending=False)
            pesan = "Berikut REKOMENDASI TERBAIK kami berdasarkan analisis:"

        # Tampilkan 5 Hasil Teratas
        print(f"AI   ➤ {pesan}")
        cols_show = ['Tempat_Makan', 'Biaya', 'Jarak_Meter', 'Rating_Rasa', 'Skor_Rekomendasi']
        print(hasil[cols_show].head(5).to_string(index=False))

# Jalankan AI Assistant
run_ai_assistant()