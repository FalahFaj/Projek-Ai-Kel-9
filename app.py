from flask import Flask, render_template, request, jsonify
import pandas as pd
import re


import model.fuzzy as fuzzy_logic
import model.kmeans as kmeans_clustering

app = Flask(__name__)


print("--- [Server] Memuat Data... ---")

try:
        # Memuat data
        df = pd.read_csv('Data/data_berlabel.csv')
    
    # Jalankan Fuzzy Logic
        df = fuzzy_logic.apply_fuzzy_scoring(df)
    
    # --- PERSIAPAN DATA UNTUK BOT & TAMPILAN ---
  
        df['Biaya_Angka'] = pd.to_numeric(df['Biaya'], errors='coerce').fillna(0)

        # 2. Format Tampilan (misal: "Rp 15,000") untuk Website
        df['Biaya_Format'] = df['Biaya_Angka'].apply(lambda x: f"Rp {int(x):,}")
        
        # 3. Bulatkan Fuzzy Score
        df['Fuzzy_Score'] = df['Fuzzy_Score'].round(1)
    
    # 4. Labeling Cluster
        def label_cluster(row):
            if row['Cluster'] == 0: return "Hemat & Enak"
            if row['Cluster'] == 1: return "Premium/Nongkrong"
            return "Standar/Harian"
        
        df['Cluster_Label'] = df.apply(label_cluster, axis=1)

        print("--- [Server] Data Siap! ---")

except Exception as e:
    print(f"Error Loading Data: {e}")
    df = pd.DataFrame()


def get_bot_response(user_msg):
    try:
        msg = user_msg.lower()
        
        # Cek jika data kosong
        if df.empty:
            return "Maaf, data sedang tidak tersedia."

        n = 3 # Default
        match = re.search(r'\d+', msg) # Cari satu atau lebih digit
        if match:
            try:
                num_requested = int(match.group(0))
                if num_requested > 0 and num_requested <= len(df): # Pastikan angkanya valid
                    n = num_requested
            except (ValueError, IndexError):
                pass # Abaikan jika tidak bisa di-parse, pakai default


        # A. User tanya rekomendasi umum
        if "rekomendasi" in msg or "saran" in msg or "makan apa" in msg:
            top_places = df.sort_values(by='Fuzzy_Score', ascending=False).head(n)
            names = top_places['Tempat_Makan'].tolist()
            return f"Halo! Berikut {len(names)} tempat rekomendasi terbaik: <b>{', '.join(names)}</b>."

        # B. User cari yang MURAH (Logic pakai Biaya_Angka)
        elif "murah" in msg or "hemat" in msg or "terjangkau" in msg:
            cheap = df[df['Biaya_Angka'] <= 15000].sort_values(by='Fuzzy_Score', ascending=False).head(n)
            names = cheap['Tempat_Makan'].tolist()
            if not names: return "Waduh, belum nemu yang murah banget nih."
            return f"Buat yang dompet mahasiswa (di bawah 15rb), ini {len(names)} rekomendasinya: <b>{', '.join(names)}</b>."

        # C. User cari yang ENAK (Rating Rasa)
        elif "enak" in msg or "lezat" in msg or "rasa" in msg:
            tasty = df[df['Rating_Rasa'] >= 4.5].sort_values(by='Fuzzy_Score', ascending=False).head(n)
            names = tasty['Tempat_Makan'].tolist()
            if not names: return "Tidak ada tempat dengan rating rasa di atas 4.5 saat ini."
            return f"Kalau cari rasa juara (Rating tinggi), ini {len(names)} rekomendasi saya: <b>{', '.join(names)}</b>."

        # D. User cari tempat NONGKRONG / WIFI
        elif "nongkrong" in msg or "wifi" in msg or "nugas" in msg:
            cozy = df[df['Cluster_Label'] == "Premium/Nongkrong"].sort_values(by='Fuzzy_Score', ascending=False).head(n)
            names = cozy['Tempat_Makan'].tolist()
            if not names: return "Belum ada data tempat nongkrong yang pas."
            return f"Buat nugas atau nongkrong nyaman, ini {len(names)} rekomendasinya: <b>{', '.join(names)}</b>."

        # E. Default Response
        else:
            return "Maaf, saya bot sederhana. Coba tanya 'rekomendasi', 'tempat murah', atau 'tempat nugas' 'harga'."

    except Exception as e:
        print(f"Bot Error: {e}")
        return "Maaf, otak saya sedang error sedikit."


@app.route('/')
def index():
    selected_cluster = request.args.get('cluster')
    data_display = df.copy()
    
    # Filter Cluster
    if selected_cluster and selected_cluster != 'All':
        data_display = data_display[data_display['Cluster_Label'] == selected_cluster]
    
    # Sorting berdasarkan Score
    data_display = data_display.sort_values(by='Fuzzy_Score', ascending=False)
    
    clusters = df['Cluster_Label'].unique()
    places = data_display.to_dict(orient='records')
    
    # Update tampilan Biaya di Tabel HTML agar pakai format "Rp ..."
    for p in places:
        p['Biaya'] = p['Biaya_Format']
    
    return render_template('index.html', places=places, clusters=clusters, selected_cluster=selected_cluster)

# API UNTUK CHAT (Dipanggil oleh Javascript)
@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    bot_reply = get_bot_response(user_input)
    return jsonify({'response': bot_reply})

if __name__ == '__main__':
    app.run(debug=True)