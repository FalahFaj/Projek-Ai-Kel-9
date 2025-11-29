import pandas as pd

def clean_data(input_path='../Data/data_dari_form.csv', output_path='../Data/data_bersih.csv'):
    """
    Membersihkan data dari file CSV mentah dan menyimpannya sebagai file CSV yang sudah bersih.

    Args:
        input_path (str): Path ke file CSV mentah.
        output_path (str): Path untuk menyimpan file CSV yang sudah bersih.
    """
    # Baca dataset
    df = pd.read_csv(input_path)

    # Ganti nama kolom
    df.columns = [
        'Timestamp', 'Nama', 'Gender', 'Tempat_Makan', 'Porsi',
        'Biaya', 'Jarak_Meter', 'Rating_Rasa', 'Rating_Nyaman', 'Wifi',
        'Colokan', 'Waktu_Tunggu', 'Kategori_Harga', 'Rekomendasi', 'Nama_warung_lainnya'
    ]

    # Ganti nilai 'Lainnya' di 'Tempat_Makan' dengan nilai dari 'Nama_warung_lainnya'
    df['Tempat_Makan'] = df.apply(
        lambda row: row['Nama_warung_lainnya'] if row['Tempat_Makan'] == 'Lainnya' else row['Tempat_Makan'],
        axis=1
    )
    
    # Hapus spasi di awal dan akhir dari 'Tempat_Makan'
    df['Tempat_Makan'] = df['Tempat_Makan'].str.strip()

    # Hapus kolom yang tidak diperlukan
    df = df.drop(columns=['Timestamp', 'Nama', 'Nama_warung_lainnya'])

    # Simpan dataframe yang sudah bersih ke file CSV
    df.to_csv(output_path, index=False)
    print(f"Data bersih telah disimpan di {output_path}")

if __name__ == '__main__':
    # Menjalankan fungsi clean_data
    clean_data(input_path='../Data/data_dari_form.csv', output_path='../Data/data_bersih.csv')