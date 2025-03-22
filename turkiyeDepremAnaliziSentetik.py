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

print("1. VERİ SETİ YÜKLEME VE İNCELEME")
print("=" * 50)

# Türkiye deprem verilerini yükle
# Gerçek uygulamada, aşağıdaki yolu değiştirmeniz gerekir
# Örnek olarak veri setini yeniden oluşturalım
np.random.seed(42)

# Veri çerçevesi oluştur
# CSV'deki örnek veri yapısına dayanarak sentetik veri oluşturalım
n_samples = 2000

# Tarih üretme fonksiyonu
def generate_dates(n, start_year=1915, end_year=2021):
    dates = []
    for _ in range(n):
        year = np.random.randint(start_year, end_year+1)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)  # Basitlik için 28 günü geçmeyelim
        dates.append(f"{year}.{month:02d}.{day:02d}")
    return dates

# Zaman üretme fonksiyonu
def generate_times(n):
    times = []
    for _ in range(n):
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        ms = np.random.randint(0, 100)
        times.append(f"{hour:02d}:{minute:02d}:{second:02d}.{ms:02d}")
    return times

# Deprem yerleri
locations = [
    "AKDENIZ", "EGE DENIZI", "MARMARA DENIZI", "KARADENIZ",
    "ISTANBUL", "IZMIR", "ANKARA", "KONYA", "ANTALYA", "ERZURUM",
    "VAN", "ELAZIG", "BINGOL", "MALATYA", "HATAY", "KAHRAMANMARAS"
]

# Deprem kodu oluştur
earthquake_codes = [f"2.{year}{np.random.randint(10000, 99999)}E+13" 
                    for year in np.random.randint(1915, 2022, n_samples)]

# Deprem Tipi (Ke: Kaydedilmiş)
types = ['Ke'] * n_samples

data = {
    'No': list(range(1, n_samples + 1)),
    'Deprem Kodu': earthquake_codes,
    'Olus tarihi': generate_dates(n_samples),
    'Olus zamani': generate_times(n_samples),
    'Enlem': np.random.uniform(36.0, 42.0, n_samples),  # Türkiye'nin enlem aralığı
    'Boylam': np.random.uniform(26.0, 45.0, n_samples),  # Türkiye'nin boylam aralığı
    'Derinlik': np.random.uniform(5.0, 100.0, n_samples),
    'xM': np.random.uniform(2.0, 7.0, n_samples),
    'MD': np.zeros(n_samples),  # Çoğunlukla 0
    'ML': np.random.uniform(2.0, 7.0, n_samples),
    'Mw': np.random.uniform(2.0, 7.0, n_samples),
    'Ms': np.zeros(n_samples),  # Çoğunlukla 0
    'Mb': np.zeros(n_samples),  # Çoğunlukla 0
    'Tip': types,
    'Yer': np.random.choice(locations, n_samples)
}

# Bazı kayıtlarda Mw, ML, MD değerleri eksik olsun
mask = np.random.random(n_samples) < 0.1
data['Mw'][mask] = 0
mask = np.random.random(n_samples) < 0.1
data['ML'][mask] = 0
mask = np.random.random(n_samples) < 0.05
data['MD'][mask] = np.random.uniform(2.0, 5.0, sum(mask))

# Veri çerçevesini oluştur
df = pd.DataFrame(data)

# Veri çerçevesini incele
print(f"Veri boyutu: {df.shape}")
print("\nİlk 5 satır:")
print(df.head().to_string())

print("\nVeri seti özeti:")
print(df.describe().to_string())

print("\nVeri türleri:")
print(df.dtypes)

print("\n2. VERİ ÖN İŞLEME")
print("=" * 50)

# Tarih ve zaman sütunlarını datetime'a dönüştür
df['Tarih'] = pd.to_datetime(df['Olus tarihi'], format='%Y.%m.%d', errors='coerce')
df['Yil'] = df['Tarih'].dt.year
df['Ay'] = df['Tarih'].dt.month

# Eksik değerleri kontrol et
print("\nEksik değer sayıları:")
print(df.isnull().sum())

# Mw, ML, xM sütunlarında 0 olanları NaN'a dönüştür (0 değerleri eksik veri demek)
for col in ['Mw', 'ML', 'Ms', 'Mb', 'MD']:
    df.loc[df[col] == 0, col] = np.nan

# Eksik değerleri tekrar kontrol et
print("\n0 değerleri NaN'a dönüştürüldükten sonra eksik değer sayıları:")
print(df.isnull().sum())

# Deprem büyüklüğü için birleşik bir ölçü oluştur
# Öncelik sırası: Mw > ML > xM > MD
df['Magnitude'] = df['Mw']
df.loc[df['Magnitude'].isna(), 'Magnitude'] = df.loc[df['Magnitude'].isna(), 'ML']
df.loc[df['Magnitude'].isna(), 'Magnitude'] = df.loc[df['Magnitude'].isna(), 'xM']
df.loc[df['Magnitude'].isna(), 'Magnitude'] = df.loc[df['Magnitude'].isna(), 'MD']

# Magnitude değeri eksik olanları ortalama ile doldur
df['Magnitude'].fillna(df['Magnitude'].mean(), inplace=True)

# Mw, ML eksik değerleri Magnitude ile doldur
df['Mw'].fillna(df['Magnitude'], inplace=True)
df['ML'].fillna(df['Magnitude'], inplace=True)

# Kategorik değişkenleri dönüştür
print("\nKategorik değişkenlerin dönüştürülmesi:")
le = LabelEncoder()
df['Yer_Encoded'] = le.fit_transform(df['Yer'])
df['Tip_Encoded'] = le.fit_transform(df['Tip'])

print(f"Eşsiz yer sayısı: {len(df['Yer'].unique())}")
print(f"Eşsiz tip sayısı: {len(df['Tip'].unique())}")

print("\n3. KEŞİFÇİ VERİ ANALİZİ")
print("=" * 50)

# Yıllara göre deprem sayısı
plt.figure(figsize=(12, 6))
year_counts = df['Yil'].value_counts().sort_index()
plt.bar(year_counts.index, year_counts.values)
plt.title('Yıllara Göre Deprem Sayısı')
plt.xlabel('Yıl')
plt.ylabel('Deprem Sayısı')
plt.grid(True, alpha=0.3)
plt.savefig('deprem_sayisi_yillara_gore.png')
print("\nYıllara göre deprem sayısı grafiği kaydedildi: 'deprem_sayisi_yillara_gore.png'")

# Derinlik ve büyüklük dağılımı
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Derinlik'], bins=30, color='blue', alpha=0.7)
plt.title('Deprem Derinliği Dağılımı')
plt.xlabel('Derinlik (km)')
plt.ylabel('Frekans')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(df['Magnitude'], bins=30, color='red', alpha=0.7)
plt.title('Deprem Büyüklüğü Dağılımı')
plt.xlabel('Büyüklük (Magnitude)')
plt.ylabel('Frekans')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('derinlik_buyukluk_dagilimi.png')
print("\nDerinlik ve büyüklük dağılımı grafiği kaydedildi: 'derinlik_buyukluk_dagilimi.png'")

# Deprem haritası (Enlem-Boylam)
plt.figure(figsize=(10, 8))
plt.scatter(df['Boylam'], df['Enlem'], c=df['Magnitude'], 
            cmap='YlOrRd', alpha=0.6, s=df['Magnitude']**2)
plt.colorbar(label='Deprem Büyüklüğü')
plt.title('Türkiye Deprem Haritası (1915-2021)')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.grid(True, alpha=0.3)
plt.savefig('turkiye_deprem_haritasi.png')
print("\nTürkiye deprem haritası kaydedildi: 'turkiye_deprem_haritasi.png'")

# Korelasyon matrisi
numeric_cols = ['Enlem', 'Boylam', 'Derinlik', 'xM', 'ML', 'Mw', 'Magnitude', 'Yil']
correlation = df[numeric_cols].corr()
print("\nKorelasyon Matrisi:")
print(correlation.to_string())

plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Deprem Özellikleri Korelasyon Matrisi')
plt.tight_layout()
plt.savefig('deprem_korelasyon_matrisi.png')
print("\nKorelasyon matrisi kaydedildi: 'deprem_korelasyon_matrisi.png'")

print("\n4. NORMALİZASYON VE DÖNÜŞÜM ANALİZİ")
print("=" * 50)

# D'Agostino K^2 testi fonksiyonu
def dagostino_test(data, alpha=0.5):
    """
    D'Agostino K^2 testi ile normallik kontrolü
    p değeri < alpha ise veri normal dağılım göstermiyor demektir.
    """
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
numerical_cols = ['Derinlik', 'Magnitude', 'Mw', 'ML', 'xM']

# Her bir sayısal sütun için dönüşümleri karşılaştır
for col in numerical_cols:
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
print("\nDönüşüm Karşılaştırma Tablosu:")
print(transformation_df.to_string(index=False))

# Dönüşüm karşılaştırmasını görselleştir
for col in numerical_cols[:2]:  # İlk iki sütun için
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
X = df[['Derinlik']]
y = df['Magnitude']

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
features = ['Derinlik', 'Enlem', 'Boylam', 'Yil']
X_multi = df[features]
y = df['Magnitude']

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
plt.barh(feature_importance['Özellik'], feature_importance['Önem'])
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

improvement_multi = ((r2_multi - r2) / r2) * 100 if r2 > 0 else float('inf')
improvement_rf = ((r2_rf - r2_multi) / r2_multi) * 100 if r2_multi > 0 else float('inf')

print(f"Basit'ten Çoklu'ya İyileştirme: {improvement_multi:.2f}%")
print(f"Çoklu'dan Random Forest'a İyileştirme: {improvement_rf:.2f}%")

# Model performanslarını görselleştir
models = ['Simple Linear', 'Multiple Linear', 'Random Forest']
r2_scores = [r2, r2_multi, r2_rf]
rmse_scores = [rmse, rmse_multi, rmse_rf]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(models, r2_scores, color=['blue', 'green', 'red'])
plt.title('Model Karşılaştırması - R² Değerleri')
plt.ylabel('R² Değeri')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(models, rmse_scores, color=['blue', 'green', 'red'])
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
X_cluster = df[cluster_features].copy()

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
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Kümeleri görselleştir
plt.figure(figsize=(10, 8))
for cluster in range(3):
    plt.scatter(
        df[df['Cluster'] == cluster]['Derinlik'],
        df[df['Cluster'] == cluster]['Magnitude'],
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
cluster_stats = df.groupby('Cluster').agg({
    'Magnitude': ['mean', 'min', 'max', 'count'],
    'Derinlik': ['mean', 'min', 'max']
})

print("\nKüme İstatistikleri:")
print(cluster_stats.to_string())

print("\n10. MANUEL TAHMİN ÖRNEĞİ")
print("=" * 50)

# Örnek değerler
derinlik = 15.0  # km
enlem = 38.5     # derece
boylam = 29.0    # derece
yil = 2022       # yıl

# Basit model ile tahmin
simple_prediction = simple_model.intercept_ + simple_model.coef_[0] * derinlik
print(f"Basit Model Tahmini (Derinlik = {derinlik} km):")
print(f"Tahmini Deprem Büyüklüğü: {simple_prediction:.2f}")

# Çoklu model ile tahmin
sample_features = np.array([[derinlik, enlem, boylam, yil]])
multi_prediction = multi_model.predict(sample_features)[0]
print(f"\nÇoklu Model Tahmini (Derinlik = {derinlik} km, Enlem = {enlem}, Boylam = {boylam}, Yıl = {yil}):")
print(f"Tahmini Deprem Büyüklüğü: {multi_prediction:.2f}")

# Random Forest model ile tahmin
rf_prediction = rf_model.predict(sample_features)[0]
print(f"\nRandom Forest Model Tahmini (Derinlik = {derinlik} km, Enlem = {enlem}, Boylam = {boylam}, Yıl = {yil}):")
print(f"Tahmini Deprem Büyüklüğü: {rf_prediction:.2f}")

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