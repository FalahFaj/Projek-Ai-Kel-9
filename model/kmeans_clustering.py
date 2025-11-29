# ==========================================
# FILE: model/kmeans_clustering.py
# ==========================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- PENGATURAN PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_folder = os.path.join(project_root, 'Data')

input_file = os.path.join(data_folder, 'data_mentah.csv')
output_file = os.path.join(data_folder, 'data_berlabel.csv')

print(f"[INFO] Membaca data dari: {input_file}")

# 1. LOAD DATA
if not os.path.exists(input_file):
    print(f"ERROR: File {input_file} tidak ditemukan!")
    exit(1)

df = pd.read_csv(input_file)

# 2. PREPROCESSING
wifi_map = {'Tidak ada': 0, 'Ada tapi lemot': 1, 'Ada dan cepat': 2}
colokan_map = {'Tidak ada': 0, 'Ada beberapa': 1, 'Ada banyak': 2}
waktu_map = {'Lama (> 20 menit)': 0, 'Sedang (10-20 menit)': 1, 'Cepat (< 10 menit)': 2}

df['Wifi_Score'] = df['Wifi'].map(wifi_map).fillna(0)
df['Colokan_Score'] = df['Colokan'].map(colokan_map).fillna(0)
df['Waktu_Score'] = df['Waktu_Tunggu'].map(waktu_map).fillna(0)

# Agregasi
data_places = df.groupby('Tempat_Makan').agg({
    'Biaya': 'mean',
    'Jarak_Meter': 'mean',
    'Rating_Rasa': 'mean',
    'Rating_Nyaman': 'mean',
    'Wifi_Score': 'mean',
    'Colokan_Score': 'mean',
    'Waktu_Score': 'mean'
}).reset_index()

print(f"[INFO] Ditemukan {len(data_places)} tempat makan unik.")

# 3. CLUSTERING
features = ['Biaya', 'Jarak_Meter', 'Rating_Rasa']
X = data_places[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data_places['Cluster'] = kmeans.fit_predict(X_scaled)

print("\n[INFO] Rata-rata per Cluster:")
print(data_places.groupby('Cluster')[features].mean())

# 4. VISUALISASI (LANGSUNG SIMPAN PNG)
sns.set_style("whitegrid")

# Plot 1: Biaya vs Rasa
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_places, x='Biaya', y='Rating_Rasa', hue='Cluster', palette='viridis', s=100)
plt.title('Clustering: Harga vs Rasa')
plot1_path = os.path.join(data_folder, 'cluster_harga_rasa.png')
plt.savefig(plot1_path)
plt.close() # Menutup plot agar tidak memakan memori/popup
print(f"[INFO] Grafik 1 disimpan ke: {plot1_path}")

# Plot 2: Jarak vs Rasa
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_places, x='Jarak_Meter', y='Rating_Rasa', hue='Cluster', palette='coolwarm', s=100)
plt.title('Clustering: Jarak vs Rasa')
plot2_path = os.path.join(data_folder, 'cluster_jarak_rasa.png')
plt.savefig(plot2_path)
plt.close() # Menutup plot
print(f"[INFO] Grafik 2 disimpan ke: {plot2_path}")

# Simpan CSV
data_places.to_csv(output_file, index=False)
print(f"[INFO] Data berlabel disimpan ke: {output_file}")