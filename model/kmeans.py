import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_clustering(df, n_clusters=3):

    print("--- [K-Means] Memulai Preprocessing & Clustering... ---")
    
    # 1. Mapping Data Kategorikal
    wifi_map = {'Tidak ada': 0, 'Ada tapi lemot': 1, 'Ada dan cepat': 2}
    colokan_map = {'Tidak ada': 0, 'Ada beberapa': 1, 'Ada banyak': 2}
    waktu_map = {
        'Lama (> 20 menit)': 0, 
        'Sedang (10-20 menit)': 1, 
        'Cepat (< 10 menit)': 2
    }

    if 'Wifi' in df.columns:
        df['Wifi_Score'] = df['Wifi'].astype(str).map(wifi_map).fillna(0)
    if 'Colokan' in df.columns:
        df['Colokan_Score'] = df['Colokan'].astype(str).map(colokan_map).fillna(0)
    if 'Waktu_Tunggu' in df.columns:
        df['Waktu_Score'] = df['Waktu_Tunggu'].astype(str).map(waktu_map).fillna(0)
    
    # 2. Cleaning Numeric Columns
    cols_to_clean = ['Biaya', 'Jarak_Meter', 'Rating_Rasa', 'Rating_Nyaman']
    for col in cols_to_clean:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Agregasi: Mengelompokkan berdasarkan TEMPAT MAKAN
    # Kita ingin tahu karakteristik "Tempat Makan"-nya, bukan per transaksi orang
    feature_cols = ['Biaya', 'Jarak_Meter', 'Rating_Rasa', 'Rating_Nyaman', 
                    'Wifi_Score', 'Colokan_Score', 'Waktu_Score']
    
    # Ambil rata-rata karakteristik setiap tempat makan
    data_places = df.groupby('Tempat_Makan')[feature_cols].mean().reset_index()
    
    # 4. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_places[feature_cols])
    
    # 5. K-Means Algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data_places['Cluster'] = kmeans.fit_predict(X_scaled)
    
    print(f"--- [K-Means] Selesai. {len(data_places)} tempat makan dikelompokkan. ---")
    
    return data_places