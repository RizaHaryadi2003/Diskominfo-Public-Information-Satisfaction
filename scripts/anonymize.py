import os
import glob
import pandas as pd

def anonymize_data():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    
    # Buat direktori jika belum ada
    os.makedirs(processed_dir, exist_ok=True)
    
    # Cari semua file CSV di folder raw
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    if not csv_files:
        print("[-] Tidak ada file CSV ditemukan di folder data/raw/")
        return

    # Gabungkan semua data dari CSV yang ada
    all_data = []
    
    for file in csv_files:
        print(f"[*] Membaca {os.path.basename(file)}...")
        # Coba baca dengan koma (format web scraper baru)
        df = pd.read_csv(file, sep=",", on_bad_lines="skip")
        
        # Mapping kolom format web scraper ke format model
        # Asumsi format scraper: web_scraper_order, ..., nama_lengkap, pendidikan, pekerjaan, phone, waktu, index, data2(Q1), usia, jenis_kelamin, data3(Q2) ...
        # Periksa apakah ini format baru dari scraper
        if "nama_lengkap" in df.columns:
            # Re-order kolom agar sesuai dengan kebutuhan api/model.py
            expected_cols = [
                "nama_lengkap", "pendidikan", "pekerjaan", "phone", "waktu",
                "index", "usia", "jenis_kelamin",
                "data2", "data3", "data4", "data5", "data6", "data7", 
                "data8", "data9", "data10", "q10", "q11", "q12", "q13", "q14", "q15"
            ]
            
            # Pastikan semua kolom yang dibutuhkan ada
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f"    [!] Melewati {os.path.basename(file)} karena tidak ada kolom: {missing}")
                continue
                
            df_selected = df[expected_cols].copy()
            
        else:
            # Jika ini format lama (mungkin dipisahkan titik koma)
            df = pd.read_csv(file, sep=";", on_bad_lines="skip")
            if "Nama" in df.columns:
                expected_cols = [
                    "Nama", "Pendidikan", "Pekerjaan", "Kontak", "Waktu",
                    "Index", "Usia", "Jenis_Kelamin",
                    "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7",
                    "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"
                ]
                df_selected = df[expected_cols].copy()
            else:
                print(f"    [!] Melewati {os.path.basename(file)} karena format tidak dikenali.")
                continue

        # Ganti nama kolom ke format standar agar konsisten
        df_selected.columns = [
            "Nama", "Pendidikan", "Pekerjaan", "Kontak", "Waktu",
            "Index", "Usia", "Jenis_Kelamin",
            "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7",
            "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"
        ]
        
        all_data.append(df_selected)

    if not all_data:
        print("[-] Tidak ada data yang berhasil diproses.")
        return

    # Gabungkan menjadi satu dataframe besar
    final_df = pd.concat(all_data, ignore_index=True)
    
    # ANONYMIZATION: Hapus identitas pribadi
    print(f"[*] Menyamarkan data privasi dari {len(final_df)} responden...")
    
    # Mengganti nama menjadi Responden_1, Responden_2, dst.
    final_df["Nama"] = [f"Responden_{i+1}" for i in range(len(final_df))]
    
    # Mengganti nomor HP/Kontak menjadi tersamarkan
    final_df["Kontak"] = "08XX-XXXX-XXXX"

    # Simpan ke folder processed sebagai satu file utama
    out_path = os.path.join(processed_dir, "Data-Aman.csv")
    final_df.to_csv(out_path, sep=";", index=False)
    
    print(f"[+] Selesai! Data aman berhasil disimpan ke: {out_path}")

if __name__ == "__main__":
    anonymize_data()
