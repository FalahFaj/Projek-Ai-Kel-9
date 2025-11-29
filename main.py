# ==========================================
# 1. INSTALASI & IMPORT LIBRARY
# ==========================================
# !pip install scikit-fuzzy

from networkx import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==========================================
# 2. LOAD & PREPROCESSING DATA
# ==========================================
# Ganti nama file sesuai file csv yang kamu upload
file_path = 'data.xlsx'
df = pd.read_excel(file_path)

df.columns = df.columns.str.strip()

# Rename kolom (Gunakan nama yang sudah bersih tanpa spasi di ujung)
df.rename(columns={
    'Nama tempat makan': 'Nama_Awal',
    'Nama warung (Jika memilih lainnya)': 'Nama_Lainnya',
    'Berapa rata-rata biaya yang kamu habiskan untuk sekali makan + minum? (Tulis angka saja, misal: 15000)': 'Harga',
    'Jarak dari Fasilkom ke tempat itu (tulis dalam meter, misal 5000)': 'Jarak',
    'Seberapa enak makanannya menurut kamu?': 'Rasa',
    'Seberapa nyaman tempat ini untuk nongkrong lama?': 'Nyaman',
    'Adakah wifinya?': 'Wifi',
    'Ketersediaan Colokan Listrik': 'Colokan',
    'Berapa lama rata-rata makanan datang setelah dipesan?': 'Waktu_Saji'
}, inplace=True)

# 2.1 Gabungkan Nama Tempat (Handling 'Lainnya')
df['Tempat_Makan'] = df.apply(lambda x: x['Nama_Lainnya'] if x['Nama_Awal'] == 'Lainnya' else x['Nama_Awal'], axis=1)

# Normalisasi Nama (Huruf kecil semua agar 'kulos' dan 'Kulos' dianggap sama)
df['Tempat_Makan'] = df['Tempat_Makan'].str.title().str.strip()

# 2.2 Encoding Kategorikal ke Numerik untuk perhitungan
# Wifi: Tidak ada=0, Lemot=1, Cepat=2
wifi_map = {'Tidak ada': 0, 'Ada tapi lemot': 1, 'Ada dan cepat': 2}
# Colokan: Tidak ada=0, Beberapa=1, Banyak=2
colokan_map = {'Tidak ada': 0, 'Ada beberapa': 1, 'Ada banyak': 2}
# Waktu Saji: Lama=0, Sedang=1, Cepat=2
waktu_map = {'Lama (> 20 menit)': 0, 'Sedang (10-20 menit)': 1, 'Cepat (< 10 menit)': 2}

# Menggunakan map dan fillna(0) untuk menghindari error jika ada data kosong
df['Wifi_Score'] = df['Wifi'].map(wifi_map).fillna(0)
df['Colokan_Score'] = df['Colokan'].map(colokan_map).fillna(0)
df['Waktu_Score'] = df['Waktu_Saji'].map(waktu_map).fillna(0)

# 2.3 Agregasi Data (Rata-rata per Tempat Makan)
data_places = df.groupby('Tempat_Makan').agg({
    'Harga': 'mean',
    'Jarak': 'mean',
    'Rasa': 'mean',
    'Nyaman': 'mean',
    'Wifi_Score': 'mean',
    'Colokan_Score': 'mean',
    'Waktu_Score': 'mean'
}).reset_index()

print(f"Data berhasil diproses! Ditemukan {len(data_places)} tempat makan unik.")
print(data_places.head())

# ==========================================
# 3. K-MEANS CLUSTERING (Segmentasi)
# ==========================================
# Kita gunakan fitur Harga, Jarak, dan Rasa untuk pengelompokan
features_cluster = ['Harga', 'Jarak', 'Rasa']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_places[features_cluster])

# Menentukan 3 Cluster (Misal: Budget, Standard, Premium/Jauh)
kmeans = KMeans(n_clusters=3, random_state=42)
data_places['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualisasi Cluster
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data_places, x='Harga', y='Rasa', hue='Cluster', palette='viridis', s=100)
plt.title('Segmentasi Tempat Makan (Harga vs Rasa)')
plt.show()

# ==========================================
# 4. FUZZY LOGIC SYSTEM (Sistem Rekomendasi)
# ==========================================
# Mendefinisikan Variabel Fuzzy
# Input
harga = ctrl.Antecedent(np.arange(0, 50000, 1000), 'harga')
jarak = ctrl.Antecedent(np.arange(0, 5000, 100), 'jarak')
rasa = ctrl.Antecedent(np.arange(0, 6, 0.1), 'rasa') # Skala 1-5
# Output
rekomendasi = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')

# Membership Function (Fungsi Keanggotaan)
# Harga
harga['murah'] = fuzz.trimf(harga.universe, [0, 0, 15000])
harga['sedang'] = fuzz.trimf(harga.universe, [10000, 20000, 30000])
harga['mahal'] = fuzz.trapmf(harga.universe, [20000, 35000, 50000, 50000])

# Jarak
jarak['dekat'] = fuzz.trimf(jarak.universe, [0, 0, 1000])
jarak['sedang'] = fuzz.trimf(jarak.universe, [500, 2000, 3500])
jarak['jauh'] = fuzz.trapmf(jarak.universe, [2000, 4000, 5000, 5000])

# Rasa
rasa['tidak_enak'] = fuzz.trimf(rasa.universe, [0, 0, 3])
rasa['biasa'] = fuzz.trimf(rasa.universe, [2, 3.5, 4.5])
rasa['enak'] = fuzz.trapmf(rasa.universe, [3.5, 4.5, 5, 5])

# Output Rekomendasi
rekomendasi['tidak_rekomen'] = fuzz.trimf(rekomendasi.universe, [0, 0, 50])
rekomendasi['pertimbangkan'] = fuzz.trimf(rekomendasi.universe, [30, 50, 70])
rekomendasi['sangat_rekomen'] = fuzz.trimf(rekomendasi.universe, [60, 100, 100])

# Rules (Aturan Fuzzy) - Logika Manusia
rule1 = ctrl.Rule(harga['murah'] & rasa['enak'], rekomendasi['sangat_rekomen'])
rule2 = ctrl.Rule(harga['sedang'] & rasa['enak'] & jarak['dekat'], rekomendasi['sangat_rekomen'])
rule3 = ctrl.Rule(harga['mahal'] & rasa['enak'], rekomendasi['pertimbangkan'])
rule4 = ctrl.Rule(rasa['tidak_enak'], rekomendasi['tidak_rekomen'])
rule5 = ctrl.Rule(harga['murah'] & jarak['jauh'], rekomendasi['pertimbangkan'])
rule6 = ctrl.Rule(harga['mahal'] & rasa['biasa'], rekomendasi['tidak_rekomen'])

# Membuat Control System
rekomendasi_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
rekomendasi_sim = ctrl.ControlSystemSimulation(rekomendasi_ctrl)

# ==========================================
# 5. MENJALANKAN SISTEM REKOMENDASI
# ==========================================
def hitung_skor_fuzzy(row):
    try:
        rekomendasi_sim.input['harga'] = row['Harga']
        rekomendasi_sim.input['jarak'] = row['Jarak']
        rekomendasi_sim.input['rasa'] = row['Rasa']
        rekomendasi_sim.compute()
        return rekomendasi_sim.output['rekomendasi']
    except:
        return 0

# Terapkan fuzzy logic ke setiap tempat makan
data_places['Skor_Rekomendasi'] = data_places.apply(hitung_skor_fuzzy, axis=1)

# Urutkan berdasarkan Skor Tertinggi
top_rekomendasi = data_places.sort_values(by='Skor_Rekomendasi', ascending=False)

# Tampilkan Hasil Akhir
print("=== HASIL REKOMENDASI TEMPAT MAKAN (Top 10) ===")
print(top_rekomendasi[['Tempat_Makan', 'Harga', 'Jarak', 'Rasa', 'Cluster', 'Skor_Rekomendasi']].head(10))

# Export hasil ke CSV jika perlu untuk laporan
# top_rekomendasi.to_csv('Hasil_Analisis_Rekomendasi.csv', index=False)