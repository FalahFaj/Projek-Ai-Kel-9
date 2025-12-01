import pandas as pd
import os

# --- PENGATURAN PATH OTOMATIS ---
# Mendapatkan path absolut ke direktori tempat script ini berada (yaitu, .../model)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Mendapatkan path ke direktori root proyek (satu tingkat di atas /model)
project_root = os.path.dirname(script_dir)
# Membuat path absolut ke direktori Data
data_dir = os.path.join(project_root, 'Data')


def clean_data(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"ERROR: File input tidak ditemukan di path: {input_path}")
        print("Pastikan file 'data_mentah.csv' ada di dalam direktori 'Data'.")
        return

    # Hapus spasi di awal dan akhir dari 'Tempat_Makan'
    if 'Tempat_Makan' in df.columns:
        df['Tempat_Makan'] = df['Tempat_Makan'].str.strip()

    # Simpan dataframe yang sudah bersih ke file CSV
    df.to_csv(output_path, index=False)
    print(f"Data bersih telah disimpan di {output_path}")

if __name__ == '__main__':
    # Menentukan path input dan output secara dinamis
    input_file = os.path.join(data_dir, 'data_mentah.csv')
    output_file = os.path.join(data_dir, 'data_bersih.csv')
    
    # Menjalankan fungsi clean_data
    clean_data(input_path=input_file, output_path=output_file)