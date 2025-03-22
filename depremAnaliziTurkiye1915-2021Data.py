import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import boxcox, normaltest, zscore
import math
import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Matplotlib Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

print("1. VERİ SETİ YÜKLEME VE İNCELEME")
print("=" * 50)

# Türkiye deprem verilerini yükle
file_path = 'turkey_earthquakes(1915-2021).csv'  # Dosya yolu

# CSV dosyasını noktalı virgül ayırıcı ile oku ve ilk satırı başlık olarak kullan
try:
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    print("UTF-8 kodlaması ile dosya başarıyla okundu.")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, sep=';', encoding='latin1')
        print("Latin-1 kodlaması ile dosya başarıyla okundu.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, sep=';', encoding='cp1254')  # Türkçe Windows kodlaması
            print("CP1254 kodlaması ile dosya başarıyla okundu.")
        except:
            print("Dosya okuma hatası! Lütfen dosya yolunu kontrol edin.")
            raise

# Veri çerçevesini incele
print(f"Veri boyutu: {df.shape}")
print("\nİlk 5 satır:")
print(df.head().to_string())

print("\nSütun isimleri:")
print(df.columns.tolist())

print("\nVeri seti özeti:")
print(df.describe().to_string())

print("\nVeri türleri:")
print(df.dtypes)

print("\n2. VERİ ÖN İŞLEME")
print("=" * 50)

# Tarih ve zaman sütunlarını datetime'a dönüştür
try:
    df['Olus tarihi'] = df['Olus tarihi'].astype(str)
    df['Tarih'] = pd.to_datetime(df['Olus tarihi'], format='%Y.%m.%d', errors='coerce')
    df['Yil'] = df['Tarih'].dt.year
    df['Ay'] = df['Tarih'].dt.month
    print("Tarih sütunları başarıyla oluşturuldu.")
except KeyError:
    print("'Olus tarihi' sütunu bulunamadı. Sütun isimlerini kontrol edin.")
    # Alternatif sütun isimlerini dene
    date_column = [col for col in df.columns if 'tarih' in col.lower() or 'date' in col.lower()]
    if date_column:
        df['Tarih'] = pd.to_datetime(df[date_column[0]], format='%Y.%m.%d', errors='coerce')
        df['Yil'] = df['Tarih'].dt.year
        df['Ay'] = df['Tarih'].dt.month
        print(f"Alternatif tarih sütunu '{date_column[0]}' kullanıldı.")
    else:
        print("Tarih sütunu bulunamadı. Yıl ve ay analizleri atlanacak.")
        df['Yil'] = 0
        df['Ay'] = 0

# Gerekli sütunların varlığını kontrol et ve isimlerini düzelt
numerical_columns = []
for expected, alternatives in [
    ('Derinlik', ['Derinlik', 'Depth', 'depth']),
    ('Enlem', ['Enlem', 'Latitude', 'latitude', 'lat']),
    ('Boylam', ['Boylam', 'Longitude', 'longitude', 'lon']),
    ('xM', ['xM', 'Magnitude', 'magnitude', 'mag']),
    ('ML', ['ML', 'LocalMagnitude']),
    ('Mw', ['Mw', 'MomentMagnitude']),
    ('MD', ['MD', 'DurationMagnitude']),
    ('Ms', ['Ms', 'SurfaceMagnitude']),
    ('Mb', ['Mb', 'BodyMagnitude']),
]:
    found = False
    for alt in alternatives:
        if alt in df.columns:
            if alt != expected:
                df[expected] = df[alt]
            numerical_columns.append(expected)
            found = True
            break
    if not found:
        print(f"'{expected}' sütunu ve alternatifleri bulunamadı.")
        df[expected] = np.nan  # Eksik sütunları NaN ile doldur

# Eksik değerleri kontrol et
print("\nEksik değer sayıları:")
print(df.isnull().sum())

# Mw, ML, xM sütunlarında 0 olanları NaN'a dönüştür (0 değerleri eksik veri demek)
for col in ['Mw', 'ML', 'Ms', 'Mb', 'MD', 'xM']:
    if col in df.columns:
        df.loc[df[col] == 0, col] = np.nan

# Eksik değerleri tekrar kontrol et
print("\n0 değerleri NaN'a dönüştürüldükten sonra eksik değer sayıları:")
print(df.isnull().sum())

# Deprem büyüklüğü için birleşik bir ölçü oluştur
# Öncelik sırası: Mw > ML > xM > MD
df['Magnitude'] = np.nan
for col in ['Mw', 'ML', 'xM', 'MD']:
    if col in df.columns:
        mask = df['Magnitude'].isna()
        df.loc[mask, 'Magnitude'] = df.loc[mask, col]

# Magnitude değeri eksik olanları ortalama ile doldur
mean_magnitude = df['Magnitude'].mean()
df['Magnitude'].fillna(mean_magnitude, inplace=True)
print(f"\nOrtalama deprem büyüklüğü: {mean_magnitude:.2f}")

# Yer bilgisi için işlemler
location_column = [col for col in df.columns if 'yer' in col.lower() or 'location' in col.lower() or 'place' in col.lower()]
if location_column:
    loc_col = location_column[0]
    print(f"\nEn yaygın 10 deprem lokasyonu:")
    print(df[loc_col].value_counts().head(10))
    
    # Kategorik değişkenleri dönüştür
    print("\nKategorik değişkenlerin dönüştürülmesi:")
    le = LabelEncoder()
    df['Yer_Encoded'] = le.fit_transform(df[loc_col])
    print(f"Eşsiz yer sayısı: {len(df[loc_col].unique())}")
else:
    print("\nYer bilgisi sütunu bulunamadı.")
    df['Yer_Encoded'] = 0

# Tip bilgisi için işlemler
type_column = [col for col in df.columns if 'tip' in col.lower() or 'type' in col.lower()]
if type_column:
    typ_col = type_column[0]
    print(f"\nDeprem tipleri:")
    print(df[typ_col].value_counts())
    
    # Boş ve NaN değerleri en yaygın değerle doldur
    most_common = df[typ_col].mode()[0]
    df[typ_col].fillna(most_common, inplace=True)
    df.loc[df[typ_col] == '', typ_col] = most_common
    
    # Kategorik değişkenleri dönüştür
    le = LabelEncoder()
    df['Tip_Encoded'] = le.fit_transform(df[typ_col])
    print(f"Eşsiz tip sayısı: {len(df[typ_col].unique())}")
else:
    print("\nTip bilgisi sütunu bulunamadı.")
    df['Tip_Encoded'] = 0

print("\n3. KEŞİFÇİ VERİ ANALİZİ")
print("=" * 50)

# Büyük depremleri filtrele
big_earthquakes = df[df['Magnitude'] >= 5.0].copy()
print(f"Büyüklüğü 5.0 ve üzeri deprem sayısı: {len(big_earthquakes)}")

# Yıllara göre deprem sayısı - sadece veri olan yıllar
if df['Yil'].max() > 0:
    valid_years = df[df['Yil'] > 1900]['Yil']
    plt.figure(figsize=(15, 6))
    
    year_counts = valid_years.value_counts().sort_index()
    plt.bar(year_counts.index, year_counts.values, alpha=0.7)
    plt.title('Yıllara Göre Deprem Sayısı')
    plt.xlabel('Yıl')
    plt.ylabel('Deprem Sayısı')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('deprem_sayisi_yillara_gore.png')
    print("\nYıllara göre deprem sayısı grafiği kaydedildi: 'deprem_sayisi_yillara_gore.png'")
    
    # Yıllara göre büyük deprem sayısı
    big_years = big_earthquakes[big_earthquakes['Yil'] > 1900]['Yil']
    big_year_counts = big_years.value_counts().sort_index()
    
    plt.figure(figsize=(15, 6))
    plt.bar(big_year_counts.index, big_year_counts.values, color='red', alpha=0.7)
    plt.title('Yıllara Göre Büyük Deprem (M ≥ 5.0) Sayısı')
    plt.xlabel('Yıl')
    plt.ylabel('Deprem Sayısı')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('buyuk_deprem_sayisi_yillara_gore.png')
    print("\nYıllara göre büyük deprem sayısı grafiği kaydedildi: 'buyuk_deprem_sayisi_yillara_gore.png'")

# Derinlik ve büyüklük dağılımı
plt.figure(figsize=(15, 6))

# Derinlik histogramı - aykırı değerler hariç
depth_data = df['Derinlik'].dropna()
depth_threshold = depth_data.quantile(0.99)  # Üst %1'i hariç tut
depth_filtered = depth_data[depth_data <= depth_threshold]

plt.subplot(1, 2, 1)
plt.hist(depth_filtered, bins=30, color='blue', alpha=0.7)
plt.title('Deprem Derinliği Dağılımı (Aykırı Değerler Hariç)')
plt.xlabel('Derinlik (km)')
plt.ylabel('Frekans')
plt.grid(True, alpha=0.3)

# Büyüklük histogramı
plt.subplot(1, 2, 2)
plt.hist(df['Magnitude'].dropna(), bins=30, color='red', alpha=0.7)
plt.title('Deprem Büyüklüğü Dağılımı')
plt.xlabel('Büyüklük (Magnitude)')
plt.ylabel('Frekans')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('derinlik_buyukluk_dagilimi.png')
print("\nDerinlik ve büyüklük dağılımı grafiği kaydedildi: 'derinlik_buyukluk_dagilimi.png'")

# Deprem haritası (Enlem-Boylam) - Türkiye sınırları içindeki ve yakın bölgedeki depremler
tr_earthquakes = df[
    (df['Enlem'] >= 35) & (df['Enlem'] <= 43) &
    (df['Boylam'] >= 25) & (df['Boylam'] <= 45) &
    (df['Magnitude'] >= 2.5)  # Anlamlı büyüklükteki depremleri göster
].copy()

plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    tr_earthquakes['Boylam'], 
    tr_earthquakes['Enlem'], 
    c=tr_earthquakes['Magnitude'], 
    cmap='YlOrRd', 
    alpha=0.6, 
    s=tr_earthquakes['Magnitude']**2,
    edgecolors='k',
    linewidths=0.3
)
plt.colorbar(scatter, label='Deprem Büyüklüğü')
plt.title('Türkiye ve Çevresi Deprem Haritası (1915-2021)')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.grid(True, alpha=0.3)

# Büyük şehirlerin yaklaşık konumları
cities = {
    'İstanbul': (29.01, 41.01),
    'Ankara': (32.85, 39.92),
    'İzmir': (27.14, 38.42),
    'Antalya': (30.71, 36.88),
    'Erzurum': (41.27, 39.90),
    'Van': (43.38, 38.50),
    'Adana': (35.32, 37.00),
    'Trabzon': (39.72, 41.00),
    'Diyarbakır': (40.23, 37.91),
    'Bursa': (29.06, 40.19)
}

for city, (lon, lat) in cities.items():
    plt.annotate(city, (lon, lat), fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.7))

plt.savefig('turkiye_deprem_haritasi.png', dpi=300, bbox_inches='tight')
print("\nTürkiye deprem haritası kaydedildi: 'turkiye_deprem_haritasi.png'")

# Büyük depremleri ayrıca göster (M >= 6.0)
major_earthquakes = df[df['Magnitude'] >= 6.0].copy()
if not major_earthquakes.empty:
    plt.figure(figsize=(12, 10))
    
    # Önce tüm depremleri göster (arka plan olarak)
    plt.scatter(
        tr_earthquakes['Boylam'], 
        tr_earthquakes['Enlem'], 
        c='gray', 
        alpha=0.2, 
        s=1
    )
    
    # Sonra büyük depremleri göster
    scatter = plt.scatter(
        major_earthquakes['Boylam'], 
        major_earthquakes['Enlem'], 
        c=major_earthquakes['Magnitude'], 
        cmap='YlOrRd', 
        alpha=0.8, 
        s=major_earthquakes['Magnitude']**2.5,
        edgecolors='k',
        linewidths=0.5
    )
    
    plt.colorbar(scatter, label='Deprem Büyüklüğü')
    plt.title('Türkiye ve Çevresi Büyük Depremler (M ≥ 6.0)')
    plt.xlabel('Boylam')
    plt.ylabel('Enlem')
    plt.grid(True, alpha=0.3)
    
    # Şehirleri göster
    for city, (lon, lat) in cities.items():
        plt.annotate(city, (lon, lat), fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.7))
    
    plt.savefig('turkiye_buyuk_depremler.png', dpi=300, bbox_inches='tight')
    print("\nTürkiye büyük depremler haritası kaydedildi: 'turkiye_buyuk_depremler.png'")

# Korelasyon matrisi
numeric_cols = ['Enlem', 'Boylam', 'Derinlik', 'Magnitude']
if 'Yil' in df.columns and df['Yil'].max() > 0:
    numeric_cols.append('Yil')

# Eksik değerleri filtrele
correlation_df = df[numeric_cols].dropna()
if len(correlation_df) > 0:
    correlation = correlation_df.corr()
    print("\nKorelasyon Matrisi:")
    print(correlation.to_string())

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Deprem Özellikleri Korelasyon Matrisi')
    plt.tight_layout()
    plt.savefig('deprem_korelasyon_matrisi.png')
    print("\nKorelasyon matrisi kaydedildi: 'deprem_korelasyon_matrisi.png'")

# Zamanla deprem büyüklükleri (son 50 yıl)
if 'Yil' in df.columns and df['Yil'].max() > 0:
    current_year = int(df['Yil'].max())
    last_50_years = df[df['Yil'] >= (current_year - 50)].copy()
    
    if not last_50_years.empty:
        plt.figure(figsize=(15, 6))
        
        # Yıllara göre maksimum deprem büyüklükleri
        yearly_max = last_50_years.groupby('Yil')['Magnitude'].max()
        
        plt.plot(yearly_max.index, yearly_max.values, 'ro-', label='Yıllık Maksimum Büyüklük')
        plt.axhline(y=7.0, color='r', linestyle='--', alpha=0.7, label='7.0 Büyüklük Eşiği')
        plt.axhline(y=6.0, color='orange', linestyle='--', alpha=0.7, label='6.0 Büyüklük Eşiği')
        plt.axhline(y=5.0, color='g', linestyle='--', alpha=0.7, label='5.0 Büyüklük Eşiği')
        
        plt.title('Son 50 Yıldaki Yıllık Maksimum Deprem Büyüklükleri')
        plt.xlabel('Yıl')
        plt.ylabel('Maksimum Deprem Büyüklüğü')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('yillik_maksimum_deprem_buyuklukleri.png')
        print("\nYıllık maksimum deprem büyüklükleri grafiği kaydedildi: 'yillik_maksimum_deprem_buyuklukleri.png'")

print("\n4. NORMALİZASYON VE DÖNÜŞÜM ANALİZİ")
print("=" * 50)

# D'Agostino K^2 testi fonksiyonu
def dagostino_test(data, alpha=0.5):
    """
    D'Agostino K^2 testi ile normallik kontrolü
    p değeri < alpha ise veri normal dağılım göstermiyor demektir.
    """
    if len(data.dropna()) < 8:  # Test için minimum 8 veri noktası gerekli
        print("Yetersiz veri - test yapılamıyor")
        return None
    
    stat, p = normaltest(data.dropna())
    print(f"İstatistik: {stat:.4f}, p-değeri: {p:.4f}")
    if p < alpha:
        print(f"p < {alpha}: Veri normal dağılım göstermiyor - Dönüşüm gerekli")
        return False
    else:
        print(f"p >= {alpha}: Veri normal dağılım gösteriyor - Dönüşüm gerekli değil")
        return True

# Dönüşüm sonuçlarını saklamak için liste
transformation_results = []

# Numerik sütunları seç
numerical_cols = ['Derinlik', 'Magnitude', 'Enlem', 'Boylam']
if 'Mw' in df.columns:
    numerical_cols.append('Mw')
if 'ML' in df.columns:
    numerical_cols.append('ML')

# Her bir sayısal sütun için dönüşümleri karşılaştır
for col in numerical_cols:
    # Bu sütunda en az 100 geçerli veri olmalı
    if df[col].count() < 100:
        print(f"\n{col} sütununda yeterli veri yok, atlanıyor.")
        continue
        
    print(f"\nDönüşüm karşılaştırması: {col}")
    print("-" * 30)
    
    # Orijinal veri
    print(f"Orijinal {col} normallik testi:")
    original_normal = dagostino_test(df[col])
    
    # Log dönüşümü
    try:
        # 0 veya negatif değerler için 1 ekleyelim
        min_val = df[col].min()
        if min_val <= 0:
            log_data = np.log1p(df[col] - min_val + 1)
        else:
            log_data = np.log(df[col])
        print(f"\nLog dönüşümü sonrası {col} normallik testi:")
        log_normal = dagostino_test(log_data)
    except:
        print("\nLog dönüşümü uygulanamadı")
        log_normal = False
    
    # Kare kök dönüşümü
    try:
        # Negatif değerler için minimum değeri ayarlayalım
        min_val = df[col].min()
        if min_val < 0:
            sqrt_data = np.sqrt(df[col] - min_val + 1)
        else:
            sqrt_data = np.sqrt(df[col])
        print(f"\nKare kök dönüşümü sonrası {col} normallik testi:")
        sqrt_normal = dagostino_test(sqrt_data)
    except:
        print("\nKare kök dönüşümü uygulanamadı")
        sqrt_normal = False
    
    # BoxCox dönüşümü (sadece pozitif değerler için)
    try:
        # BoxCox pozitif değerler gerektirir
        min_val = df[col].min()
        if min_val <= 0:
            boxcox_data, _ = boxcox(df[col] - min_val + 1)
        else:
            boxcox_data, _ = boxcox(df[col])
        print(f"\nBoxCox dönüşümü sonrası {col} normallik testi:")
        boxcox_normal = dagostino_test(pd.Series(boxcox_data))
    except:
        print("\nBoxCox dönüşümü uygulanamadı")
        boxcox_normal = False
    
    # Sonuçları tabloya ekle
    transformation_results.append({
        'Özellik': col,
        'Orijinal Normal mi?': original_normal,
        'Log Normal mi?': log_normal,
        'Sqrt Normal mi?': sqrt_normal,
        'BoxCox Normal mi?': boxcox_normal
    })

# Dönüşüm sonuçları tablosu
transformation_df = pd.DataFrame(transformation_results)
if not transformation_df.empty:
    print("\nDönüşüm Karşılaştırma Tablosu:")
    print(transformation_df.to_string(index=False))

    # Dönüşüm sonuçlarını grafik olarak göster
    plt.figure(figsize=(12, 8))
    
    # Her özellik için dönüşüm sonuçlarını göster
    features = transformation_df['Özellik'].tolist()
    normal_cols = ['Orijinal Normal mi?', 'Log Normal mi?', 'Sqrt Normal mi?', 'BoxCox Normal mi?']
    
    # Her dönüşüm türü için bar plot oluştur
    x = np.arange(len(features))
    width = 0.2
    
    # True/False/None değerlerini sayısal değerlere dönüştür
    def convert_to_numeric(val):
        if val is True:
            return 1
        elif val is False:
            return 0
        else:
            return -1  # None değeri
    
    for i, col in enumerate(normal_cols):
        values = [convert_to_numeric(transformation_df.loc[j, col]) for j in range(len(transformation_df))]
        plt.bar(x + width * (i - 1.5), values, width, label=col)
    
    plt.ylabel('Normal Dağılım (1=Evet, 0=Hayır, -1=Test Edilemedi)')
    plt.title('Farklı Dönüşümlerin Normal Dağılıma Etkisi')
    plt.xticks(x, features)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('donusum_karsilastirma_ozet.png')
    print("\nDönüşüm karşılaştırma özet grafiği kaydedildi: 'donusum_karsilastirma_ozet.png'")

# Dönüşüm karşılaştırmasını görselleştir
for col in numerical_cols[:2]:  # İlk iki sütun için
    # Bu sütunda en az 100 geçerli veri olmalı
    if df[col].count() < 100:
        continue
        
    plt.figure(figsize=(15, 10))
    
    # Orijinal veri
    plt.subplot(2, 2, 1)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Orijinal {col}')
    
    # Log dönüşümü
    try:
        plt.subplot(2, 2, 2)
        min_val = df[col].min()
        if min_val <= 0:
            log_data = np.log1p(df[col] - min_val + 1)
        else:
            log_data = np.log(df[col])
        sns.histplot(log_data.dropna(), kde=True)
        plt.title(f'Log Dönüşümü {col}')
    except:
        plt.subplot(2, 2, 2)
        plt.text(0.5, 0.5, 'Log dönüşümü uygulanamadı', 
                 horizontalalignment='center', verticalalignment='center')
    
    # Sqrt dönüşümü
    try:
        plt.subplot(2, 2, 3)
        min_val = df[col].min()
        if min_val < 0:
            sqrt_data = np.sqrt(df[col] - min_val + 1)
        else:
            sqrt_data = np.sqrt(df[col])
        sns.histplot(sqrt_data.dropna(), kde=True)
        plt.title(f'Kare Kök Dönüşümü {col}')
    except:
        plt.subplot(2, 2, 3)
        plt.text(0.5, 0.5, 'Kare kök dönüşümü uygulanamadı', 
                 horizontalalignment='center', verticalalignment='center')
    
    # BoxCox dönüşümü
    try:
        plt.subplot(2, 2, 4)
        min_val = df[col].min()
        if min_val <= 0:
            boxcox_data, _ = boxcox(df[col] - min_val + 1)
        else:
            boxcox_data, _ = boxcox(df[col])
        sns.histplot(boxcox_data, kde=True)
        plt.title(f'BoxCox Dönüşümü {col}')
    except:
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, 'BoxCox dönüşümü uygulanamadı', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'donusum_karsilastirma_{col}.png')
    print(f"\n{col} için dönüşüm karşılaştırması kaydedildi: 'donusum_karsilastirma_{col}.png'")

print("\n5. SIMPLE LINEAR REGRESSION - DEPREM BÜYÜKLÜĞÜ TAHMİNİ")
print("=" * 50)

# Derinlik ve büyüklük arasındaki ilişki için basit doğrusal regresyon
# Eksik ve aykırı değerleri filtrele
valid_data = df.dropna(subset=['Derinlik', 'Magnitude']).copy()
depth_q1 = valid_data['Derinlik'].quantile(0.01)
depth_q3 = valid_data['Derinlik'].quantile(0.99)
valid_data = valid_data[(valid_data['Derinlik'] >= depth_q1) & 
                         (valid_data['Derinlik'] <= depth_q3)]

if len(valid_data) < 100:
    print("Basit doğrusal regresyon için yeterli veri yok.")
else:
    X = valid_data[['Derinlik']]
    y = valid_data['Magnitude']

    # Veriyi eğitim ve test kümelerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Basit doğrusal regresyon modelini eğit
    simple_model = LinearRegression()
    simple_model.fit(X_train, y_train)

    # Tahminler yap
    y_pred = simple_model.predict(X_test)

    # Modeli değerlendir
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Derinlik-Büyüklük Basit Doğrusal Regresyon Sonuçları:")
    print(f"Katsayı: {simple_model.coef_[0]:.4f}")
    print(f"Kesişim: {simple_model.intercept_:.4f}")
    print(f"Ortalama Kare Hata (MSE): {mse:.4f}")
    print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.4f}")
    print(f"R-kare (R²): {r2:.4f}")

    # Basit regresyonu görselleştir
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Gerçek Değerler')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Tahminler')
    plt.title('Deprem Derinliği ve Büyüklüğü Arasındaki İlişki')
    plt.xlabel('Derinlik (km)')
    plt.ylabel('Büyüklük (Magnitude)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('derinlik_buyukluk_regresyon.png')
    print("\nDerinlik ve büyüklük regresyon grafiği kaydedildi: 'derinlik_buyukluk_regresyon.png'")

print("\n6. MULTIPLE LINEAR REGRESSION - DEPREM BÜYÜKLÜĞÜ TAHMİNİ")
print("=" * 50)

# Çoklu doğrusal regresyon için özellikleri seç
features = ['Derinlik', 'Enlem', 'Boylam']
if 'Yil' in df.columns and df['Yil'].max() > 0:
    features.append('Yil')

# Eksik değerleri filtrele
valid_data = df.dropna(subset=features + ['Magnitude']).copy()

# Her bir özellik için aykırı değerleri filtrele (1-99 persentil)
for feature in features:
    q1 = valid_data[feature].quantile(0.01)
    q3 = valid_data[feature].quantile(0.99)
    valid_data = valid_data[(valid_data[feature] >= q1) & (valid_data[feature] <= q3)]

if len(valid_data) < 100:
    print("Çoklu doğrusal regresyon için yeterli veri yok.")
else:
    X_multi = valid_data[features]
    y = valid_data['Magnitude']

    # Veriyi eğitim ve test kümelerine ayır
    X_train_multi, X_test_multi, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

    # Çoklu doğrusal regresyon modelini eğit
    multi_model = LinearRegression()
    multi_model.fit(X_train_multi, y_train)

    # Tahminler yap
    y_pred_multi = multi_model.predict(X_test_multi)

    # Modeli değerlendir
    mse_multi = mean_squared_error(y_test, y_pred_multi)
    rmse_multi = math.sqrt(mse_multi)
    r2_multi = r2_score(y_test, y_pred_multi)

    print(f"Çoklu Doğrusal Regresyon Sonuçları:")
    print("Katsayılar:")
    for i, feature in enumerate(features):
        print(f"  {feature}: {multi_model.coef_[i]:.4f}")
    print(f"Kesişim: {multi_model.intercept_:.4f}")
    print(f"Ortalama Kare Hata (MSE): {mse_multi:.4f}")
    print(f"Kök Ortalama Kare Hata (RMSE): {rmse_multi:.4f}")
    print(f"R-kare (R²): {r2_multi:.4f}")

    # Tahmin vs gerçek değerleri görselleştir
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_multi, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Çoklu Doğrusal Regresyon: Tahmin vs Gerçek Deprem Büyüklüğü')
    plt.xlabel('Gerçek Büyüklük')
    plt.ylabel('Tahmin Edilen Büyüklük')
    plt.grid(True, alpha=0.3)
    plt.savefig('coklu_regresyon_tahmin_gercek.png')
    print("\nÇoklu regresyon tahmin vs gerçek grafiği kaydedildi: 'coklu_regresyon_tahmin_gercek.png'")

print("\n7. RANDOM FOREST REGRESSOR - DEPREM BÜYÜKLÜĞÜ TAHMİNİ")
print("=" * 50)

if len(valid_data) < 100:
    print("Random Forest regresyon için yeterli veri yok.")
else:
    # Random Forest Regresyon modelini eğit
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_multi, y_train)

    # Tahminler yap
    y_pred_rf = rf_model.predict(X_test_multi)

    # Modeli değerlendir
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = math.sqrt(mse_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"Random Forest Regresyon Sonuçları:")
    print(f"Ortalama Kare Hata (MSE): {mse_rf:.4f}")
    print(f"Kök Ortalama Kare Hata (RMSE): {rmse_rf:.4f}")
    print(f"R-kare (R²): {r2_rf:.4f}")

    # Özellik önemlerini görselleştir
    feature_importance = pd.DataFrame({
        'Özellik': features,
        'Önem': rf_model.feature_importances_
    }).sort_values('Önem', ascending=False)

    print("\nÖzellik Önemleri:")
    print(feature_importance.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Önem', y='Özellik', data=feature_importance)
    plt.title('Random Forest Regresyon - Özellik Önemleri')
    plt.xlabel('Önem')
    plt.ylabel('Özellik')
    plt.tight_layout()
    plt.savefig('rf_ozellik_onemleri.png')
    print("\nRandom Forest özellik önemleri grafiği kaydedildi: 'rf_ozellik_onemleri.png'")

    print("\n8. MODEL KARŞILAŞTIRMASI")
    print("=" * 50)

    print(f"Basit Doğrusal Regresyon R²: {r2:.4f}")
    print(f"Çoklu Doğrusal Regresyon R²: {r2_multi:.4f}")
    print(f"Random Forest Regresyon R²: {r2_rf:.4f}")

    improvement_multi = ((r2_multi - r2) / abs(r2)) * 100 if r2 != 0 else float('inf')
    improvement_rf = ((r2_rf - r2_multi) / abs(r2_multi)) * 100 if r2_multi != 0 else float('inf')

    print(f"Basit'ten Çoklu'ya İyileştirme: {improvement_multi:.2f}%")
    print(f"Çoklu'dan Random Forest'a İyileştirme: {improvement_rf:.2f}%")

    # Model performanslarını görselleştir
    models = ['Simple Linear', 'Multiple Linear', 'Random Forest']
    r2_scores = [r2, r2_multi, r2_rf]
    rmse_scores = [rmse, rmse_multi, rmse_rf]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(x=models, y=r2_scores, palette=['blue', 'green', 'red'])
    plt.title('Model Karşılaştırması - R² Değerleri')
    plt.ylabel('R² Değeri')
    plt.ylim(0, max(r2_scores) * 1.1)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.barplot(x=models, y=rmse_scores, palette=['blue', 'green', 'red'])
    plt.title('Model Karşılaştırması - RMSE Değerleri')
    plt.ylabel('RMSE Değeri')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_karsilastirmasi.png')
    print("\nModel karşılaştırması grafiği kaydedildi: 'model_karsilastirmasi.png'")

print("\n9. K-MEANS CLUSTERING - DEPREM VERİLERİNİ KÜMELEME")
print("=" * 50)

# Kümeleme için özellikleri seç
cluster_features = ['Magnitude', 'Derinlik']

# Eksik ve aykırı değerleri filtrele
cluster_data = df.dropna(subset=cluster_features).copy()
for col in cluster_features:
    q1 = cluster_data[col].quantile(0.01)
    q3 = cluster_data[col].quantile(0.99)
    cluster_data = cluster_data[(cluster_data[col] >= q1) & (cluster_data[col] <= q3)]

if len(cluster_data) < 100:
    print("K-means kümeleme için yeterli veri yok.")
else:
    X_cluster = cluster_data[cluster_features].copy()

    # Verileri standartlaştır
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Optimal küme sayısını belirlemek için Elbow yöntemi
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X_scaled)
        inertia.append(model.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Elbow Yöntemi ile Optimal Küme Sayısı Belirleme')
    plt.xlabel('Küme Sayısı (k)')
    plt.ylabel('Inertia')
    plt.grid(True, alpha=0.3)
    plt.savefig('kmeans_elbow.png')
    print("\nK-means elbow grafiği kaydedildi: 'kmeans_elbow.png'")

    # 3 küme kullanalım
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Kümeleri görselleştir
    plt.figure(figsize=(10, 8))
    for cluster in range(3):
        plt.scatter(
            cluster_data[cluster_data['Cluster'] == cluster]['Derinlik'],
            cluster_data[cluster_data['Cluster'] == cluster]['Magnitude'],
            label=f'Küme {cluster}',
            alpha=0.7
        )

    # Küme merkezlerini göster
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 1], centers[:, 0], s=200, c='black', marker='X', label='Küme Merkezleri')

    plt.title('Deprem Verilerinin K-Means ile Kümelenmesi')
    plt.xlabel('Derinlik (km)')
    plt.ylabel('Büyüklük (Magnitude)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('kmeans_clustering.png')
    print("\nK-means kümeleme grafiği kaydedildi: 'kmeans_clustering.png'")

    # Küme istatistiklerini hesapla
    cluster_stats = cluster_data.groupby('Cluster').agg({
        'Magnitude': ['mean', 'min', 'max', 'count', 'std'],
        'Derinlik': ['mean', 'min', 'max', 'std']
    })

    print("\nKüme İstatistikleri:")
    print(cluster_stats.to_string())

    # Her kümenin özelliklerini görselleştir
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(x='Cluster', y='Magnitude', data=cluster_data)
    plt.title('Kümelere Göre Deprem Büyüklüğü')
    plt.xlabel('Küme')
    plt.ylabel('Büyüklük (Magnitude)')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x='Cluster', y='Derinlik', data=cluster_data)
    plt.title('Kümelere Göre Deprem Derinliği')
    plt.xlabel('Küme')
    plt.ylabel('Derinlik (km)')
    
    # Kümeler zaman içinde nasıl dağılmış
    if 'Yil' in cluster_data.columns and cluster_data['Yil'].max() > 0:
        plt.subplot(1, 3, 3)
        sns.boxplot(x='Cluster', y='Yil', data=cluster_data)
        plt.title('Kümelere Göre Yıl Dağılımı')
        plt.xlabel('Küme')
        plt.ylabel('Yıl')
    
    plt.tight_layout()
    plt.savefig('kume_ozellikleri.png')
    print("\nKüme özellikleri grafiği kaydedildi: 'kume_ozellikleri.png'")

print("\n10. MANUEL TAHMİN ÖRNEĞİ")
print("=" * 50)

# Model eğitilmiş mi kontrol et
if 'simple_model' in locals() and 'multi_model' in locals() and 'rf_model' in locals():
    # Örnek değerler
    derinlik = 15.0  # km
    enlem = 38.5     # derece
    boylam = 29.0    # derece
    yil = 2022       # yıl

    # Basit model ile tahmin
    simple_prediction = simple_model.intercept_ + simple_model.coef_[0] * derinlik
    print(f"Basit Model Tahmini (Derinlik = {derinlik} km):")
    print(f"Tahmini Deprem Büyüklüğü: {simple_prediction:.2f}")

    # Çoklu model için özellik listesini kontrol et
    if 'Yil' in features:
        sample_features = np.array([[derinlik, enlem, boylam, yil]])
    else:
        sample_features = np.array([[derinlik, enlem, boylam]])
    
    # Çoklu model ile tahmin
    multi_prediction = multi_model.predict(sample_features)[0]
    
    # Özellikleri yazdır
    feature_str = f"Derinlik = {derinlik} km, Enlem = {enlem}, Boylam = {boylam}"
    if 'Yil' in features:
        feature_str += f", Yıl = {yil}"
    
    print(f"\nÇoklu Model Tahmini ({feature_str}):")
    print(f"Tahmini Deprem Büyüklüğü: {multi_prediction:.2f}")

    # Random Forest model ile tahmin
    rf_prediction = rf_model.predict(sample_features)[0]
    print(f"\nRandom Forest Model Tahmini ({feature_str}):")
    print(f"Tahmini Deprem Büyüklüğü: {rf_prediction:.2f}")
else:
    print("Tahmin için uygun modeller bulunamadı. Veri kalitesini kontrol edin.")

print("\n11. SONUÇ")
print("=" * 50)
print("Bu çalışmada şunları gerçekleştirdik:")
print("1. Deprem veri setinin incelenmesi ve özelliklerinin anlaşılması")
print("2. Eksik değerlerin tespiti ve uygun şekilde doldurulması")
print("3. Korelasyon analizi ile deprem büyüklüğüne etki eden faktörlerin belirlenmesi")
print("4. D'Agostino K^2 testi ile normallik kontrolü ve farklı dönüşüm tekniklerinin karşılaştırılması")
print("5. Basit doğrusal regresyon ile deprem derinliği ve büyüklüğü arasındaki ilişkinin modellenmesi")
print("6. Çoklu doğrusal regresyon ile birden fazla faktörün deprem büyüklüğüne etkisinin analizi")
print("7. Random Forest regresyon ile daha karmaşık ilişkilerin modellenmesi")
print("8. Farklı modellerin karşılaştırılması ve performans değerlendirmesi")
print("9. K-means kümeleme ile depremlerin gruplandırılması")
print("10. Örnek bir deprem için büyüklük tahmini")