# ==========================================
# FILE: main.py
# ==========================================
import subprocess
import sys
import os

def main():
    print("===================================================")
    print("   SISTEM REKOMENDASI KULINER (K-Means + Fuzzy)    ")
    print("===================================================")

    # Menentukan path script secara absolut
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_kmeans = os.path.join(base_dir, 'model', 'kmeans_clustering.py')
    script_fuzzy = os.path.join(base_dir, 'model', 'fuzzy.py')
    
    # Path output untuk pengecekan
    output_csv = os.path.join(base_dir, 'Data', 'data_berlabel.csv')

    # --- LANGKAH 1: K-MEANS & VISUALISASI CLUSTER ---
    print("\n[1/2] Menjalankan K-Means Clustering & Visualisasi...")
    
    if not os.path.exists(script_kmeans):
        print("!! ERROR: Script kmeans_clustering.py tidak ditemukan di folder 'model'.")
        return

    try:
        subprocess.run([sys.executable, script_kmeans], check=True)
        print(">> SUKSES: Clustering & Visualisasi selesai.")
    except subprocess.CalledProcessError as e:
        print(f"!! ERROR: Terjadi kesalahan saat menjalankan K-Means. (Exit code: {e.returncode})")
        return

    # Cek output
    if not os.path.exists(output_csv):
        print("!! ERROR: File 'data_berlabel.csv' gagal dibuat.")
        return

    print("-" * 50)

    # --- LANGKAH 2: FUZZY LOGIC & VISUALISASI REKOMENDASI ---
    print("\n[2/2] Menjalankan Fuzzy Logic & Visualisasi...")
    
    if not os.path.exists(script_fuzzy):
        print("!! ERROR: Script fuzzy.py tidak ditemukan di folder 'model'.")
        return

    try:
        subprocess.run([sys.executable, script_fuzzy], check=True)
        print(">> SUKSES: Rekomendasi selesai.")
    except subprocess.CalledProcessError as e:
        print(f"!! ERROR: Terjadi kesalahan saat menjalankan Fuzzy Logic. (Exit code: {e.returncode})")
        return

    print("\n===================================================")
    print("           SEMUA PROSES SELESAI DIJALANKAN         ")
    print("           Cek folder 'Data' untuk melihat grafik  ")
    print("===================================================")

if __name__ == "__main__":
    main()