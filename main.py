import pandas as pd
import sys

# Import modul custom
from model.kmeans import run_clustering
from model.fuzzy import apply_fuzzy_scoring

def chat_complex_simulation(df):
    print("\n" + "="*50)
    print("   AI KULINER ASSISTANT (ADVANCED MODEL)   ")
    print("="*50)
    print("AI: Halo! Saya menggunakan algoritma Clustering & Fuzzy Logic canggih.")
    
    while True:
        print("\n" + "-"*30)
        print("Silakan masukkan preferensi Anda (ketik 'exit' untuk keluar):")
        
        try:
            # Input User
            p_harga = input("   > Preferensi Harga (murah/sedang/mahal/bebas): ").lower().strip()
            if p_harga == 'exit': break
            
            p_wifi = input("   > Butuh Wifi buat nugas? (ya/tidak/bebas): ").lower().strip()
            p_colokan = input("   > Butuh banyak colokan? (ya/tidak/bebas): ").lower().strip()
            p_rasa = input("   > Rating rasa wajib tinggi (4.5+)? (ya/tidak): ").lower().strip()
            
        except KeyboardInterrupt:
            break

        # --- FILTERING LOGIC ---
        filtered = df.copy()
        
        # Filter Harga
        if p_harga == 'murah':
            filtered = filtered[filtered['Biaya'] <= 15000]
        elif p_harga == 'sedang':
            filtered = filtered[(filtered['Biaya'] > 15000) & (filtered['Biaya'] <= 25000)]
        elif p_harga == 'mahal':
            filtered = filtered[filtered['Biaya'] > 25000]

        # Filter Fasilitas (Memanfaatkan data Wifi/Colokan yang sudah di-encode di kmeans_model)
        # Wifi_Score: 0=Tidak ada, 1=Lemot, 2=Cepat
        if p_wifi == 'ya':
            filtered = filtered[filtered['Wifi_Score'] >= 1] # Minimal ada wifi
            
        # Colokan_Score: 0=Tidak ada, 1=Beberapa, 2=Banyak
        if p_colokan == 'ya':
            filtered = filtered[filtered['Colokan_Score'] >= 1]

        # Filter Rasa
        if p_rasa == 'ya':
            filtered = filtered[filtered['Rating_Rasa'] >= 4.5]

        # --- RANKING HASIL ---
        if filtered.empty:
            print("\nAI: Maaf, tidak ditemukan tempat dengan kriteria spesifik itu.")
            print("    Coba kurangi syarat fasilitasnya ya.")
        else:
            # Urutkan berdasarkan Fuzzy Score (Kecerdasan Buatan)
            recommendations = filtered.sort_values(by='Fuzzy_Score', ascending=False).head(5)
            
            print(f"\nAI: Ditemukan {len(filtered)} opsi. Ini Top 5 rekomendasi teratas:")
            for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
                
                # Label Cluster untuk info tambahan
                if row['Cluster'] == 0: label_cluster = "Tipe A"
                elif row['Cluster'] == 1: label_cluster = "Tipe B"
                else: label_cluster = "Tipe C"
                
                # Info Fasilitas
                info_wifi = "✅Wifi" if row['Wifi_Score'] > 0 else "❌Wifi"
                info_colokan = "✅Listrik" if row['Colokan_Score'] > 0 else "❌Listrik"

                print(f"{i}. {row['Tempat_Makan']} ({label_cluster})")
                print(f"   💰 Rp{row['Biaya']:,.0f} | ⭐ Rasa: {row['Rating_Rasa']}/5 | {info_wifi} | {info_colokan}")
                print(f"   🤖 Fuzzy Score: {row['Fuzzy_Score']:.1f}/100")

def main():
    try:
        # Load Data
        df_raw = pd.read_csv('Data_Kuliner_Fasilkom_Cleaned.csv')
    except FileNotFoundError:
        print("Error: File CSV tidak ditemukan.")
        return

    # 1. Jalankan Clustering (Preprocessing otomatis di dalam sini)
    df_clustered = run_clustering(df_raw, n_clusters=3)
    
    # 2. Jalankan Fuzzy Logic (Menghitung skor cerdas)
    df_final = apply_fuzzy_scoring(df_clustered)
    
    # 3. Simpan & Mulai Chat
    df_final.to_csv('Hasil_Advanced_AI.csv', index=False)
    chat_complex_simulation(df_final)

if __name__ == "__main__":
    main()