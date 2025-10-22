# --- Import library utama ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Baca dataset ---
df = pd.read_csv(r"C:\Users\user\Downloads\AmesHousing.csv")

print("===== 5 Data Pertama =====")
print(df.head(), "\n")

# --- 2. Bersihkan simbol mata uang di kolom SalePrice (kalau ada simbol $) ---
if 'SalePrice' in df.columns:
    df['SalePrice'] = df['SalePrice'].replace(r'[\$,]', '', regex=True).astype(float)

# --- 3. Cek tipe data dan jumlah missing values ---
print("===== Informasi Dataset =====")
df.info()
print("\n===== Jumlah Missing Values (10 Teratas) =====")
print(df.isnull().sum().sort_values(ascending=False).head(10), "\n")

# --- 4. Penanganan Null Values ---
# Kolom numerik diisi median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Kolom kategorikal diisi modus
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Setelah diisi, tampilkan info null lagi
print("===== Setelah Penanganan Null Values =====")
print(df.isnull().sum().sort_values(ascending=False).head(10), "\n")

# --- 5. Hapus data duplikat (jika ada) ---
df.drop_duplicates(inplace=True)

# --- 6. Ubah teks jadi lowercase untuk kolom tertentu (opsional) ---
if 'Neighborhood' in df.columns:
    df['Neighborhood'] = df['Neighborhood'].str.lower()

# --- 7. Statistika Deskriptif (Mean, Median, Modus, Varian, Standar Deviasi) ---
print("===== Statistika Deskriptif (SalePrice) =====")
mean_val = df['SalePrice'].mean()
median_val = df['SalePrice'].median()
mode_val = df['SalePrice'].mode()[0]
var_val = df['SalePrice'].var()
std_val = df['SalePrice'].std()

print(f"Mean (rata-rata): {mean_val:,.2f}")
print(f"Median (nilai tengah): {median_val:,.2f}")
print(f"Modus (paling sering): {mode_val:,.2f}")
print(f"Varian: {var_val:,.2f}")
print(f"Standar Deviasi: {std_val:,.2f}\n")

# --- 8. Deteksi Outlier pada kolom SalePrice (Metode IQR) ---
print("===== Deteksi Outlier pada SalePrice =====")
Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['SalePrice'] < lower_bound) | (df['SalePrice'] > upper_bound)]

print(f"Q1: {Q1:,.2f}")
print(f"Q3: {Q3:,.2f}")
print(f"IQR: {IQR:,.2f}")
print(f"Batas bawah: {lower_bound:,.2f}")
print(f"Batas atas: {upper_bound:,.2f}")
print(f"Jumlah data outlier: {len(outliers)}")

if len(outliers) > 0:
    print("\n===== Contoh Data Outlier =====")
    print(outliers[['SalePrice']].head())
else:
    print("\nTidak ada data outlier pada kolom SalePrice.")

# --- 9. Visualisasi Boxplot untuk Outlier ---
plt.figure(figsize=(6,4))
plt.boxplot(df['SalePrice'])
plt.title("Boxplot - Deteksi Outlier pada SalePrice")
plt.ylabel("Sale Price")
plt.show()

# --- 10. Simpan dataset bersih ---
df.to_csv(r"C:\Users\user\Downloads\AmesHousing_Cleaned.csv", index=False)
print("\nDataset bersih telah disimpan sebagai 'AmesHousing_Cleaned.csv'")
