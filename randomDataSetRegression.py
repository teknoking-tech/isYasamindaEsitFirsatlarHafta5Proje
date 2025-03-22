import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from scipy.stats import boxcox, normaltest
import math
import scipy.stats as stats

# Veri setini yükle (otomobil yakıt tüketimi ve emisyon verileri)
# Sentetik veri oluştur
np.random.seed(42)
n_samples = 500

# Sentetik veri oluştur
data = {
    'MAKE': np.random.choice(['Toyota', 'Honda', 'Ford', 'Hyundai', 'BMW'], n_samples),
    'MODEL': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], n_samples),
    'VEHICLE_CLASS': np.random.choice(['SUV', 'SEDAN', 'COMPACT', 'MIDSIZE', 'TRUCK'], n_samples),
    'ENGINESIZE': np.random.uniform(1.0, 5.0, n_samples),
    'CYLINDERS': np.random.choice([3, 4, 6, 8], n_samples),
    'TRANSMISSION': np.random.choice(['A', 'M', 'AM', 'AS', 'AV'], n_samples),
    'FUELTYPE': np.random.choice(['X', 'Z', 'D', 'E'], n_samples),
    'FUELCONSUMPTION_CITY': np.random.uniform(5.0, 20.0, n_samples),
    'FUELCONSUMPTION_HWY': np.random.uniform(4.0, 15.0, n_samples),
}

# FUELCONSUMPTION_COMB hesapla (şehir ve otoyol ortalaması)
data['FUELCONSUMPTION_COMB'] = (data['FUELCONSUMPTION_CITY'] + data['FUELCONSUMPTION_HWY']) / 2

# CO2EMISSIONS hesapla (yakıt tüketimi ve motor boyutu ilişkisi)
base_co2 = 100
data['CO2EMISSIONS'] = (
    base_co2 + 
    data['ENGINESIZE'] * 30 + 
    data['CYLINDERS'] * 5 + 
    data['FUELCONSUMPTION_COMB'] * 15 + 
    np.random.normal(0, 20, n_samples)  # rastgele gürültü ekle
)

# Veri çerçevesini oluştur
df = pd.DataFrame(data)

# 1. MISSING VALUE ekleme (rastgele %5 veriyi NaN yap)
mask = np.random.random(df.shape) < 0.05
df = df.mask(mask)

print("1. VERİ SETİ GENEL BAKIŞ")
print("=" * 50)
print(f"Veri seti boyutu: {df.shape}")
print("\nİlk 5 satır:")
print(df.head().to_string())

print("\n2. MISSING VALUE DETECTION")
print("=" * 50)
print("Eksik değer sayıları:")
print(df.isnull().sum())
print(f"\nToplam eksik değer: {df.isnull().sum().sum()}")

print("\n3. MISSING VALUE HANDLING")
print("=" * 50)
# Sayısal değerleri ortalama ile doldur
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Kategorik değerleri mod ile doldur
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Eksik değerler doldurulduktan sonra kalan eksik değer sayısı:")
print(df.isnull().sum().sum())

print("\n4. CATEGORICAL DEĞERLERİ DÖNÜŞTÜRME")
print("=" * 50)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Kategorik sütunlar: {categorical_columns}")

# One-Hot Encoding uygula
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("\nDönüşüm sonrası veri seti başlığı:")
print(df.head().to_string())

print("\n5. KORELASYON ANALİZİ")
print("=" * 50)
# CO2EMISSIONS ile diğer özellikler arasındaki korelasyonu hesapla
correlation = df.corr()['CO2EMISSIONS'].sort_values(ascending=False)
print("CO2EMISSIONS ile korelasyon (büyükten küçüğe):")
print(correlation)

# Korelasyon matrisini görselleştir
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.tight_layout()
plt.savefig('korelasyon_matrisi.png')
print("\nKorelasyon matrisi 'korelasyon_matrisi.png' olarak kaydedildi.")

print("\n6. NORMALİZASYON VE DÖNÜŞÜM")
print("=" * 50)

# D'Agostino K^2 testi fonksiyonu
def dagostino_test(data, alpha=0.05):
    """
    D'Agostino K^2 testi ile normallik kontrolü
    p değeri < 0.5 ise veri normal dağılım göstermiyor demektir.
    """
    stat, p = stats.normaltest(data.dropna())
    print(f"İstatistik: {stat:.4f}, p-değeri: {p:.4f}")
    if p < 0.5:
        print("p < 0.5: Veri normal dağılım göstermiyor - Dönüşüm gerekli")
        return False
    else:
        print("p >= 0.5: Veri normal dağılım gösteriyor - Dönüşüm gerekli değil")
        return True

# Dönüşüm karşılaştırma tablosu
transformation_results = []

# Sayısal sütunları seç
numerical_cols = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
                  'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']

# Her bir sayısal sütun için dönüşümleri karşılaştır
for col in numerical_cols:
    print(f"\nDönüşüm karşılaştırması: {col}")
    print("-" * 30)
    
    # Orijinal veri
    print(f"Orijinal {col} normallik testi:")
    original_normal = dagostino_test(df[col])
    original_skewness = df[col].skew()
    
    # Log dönüşümü (log(x+1) negatif değerlerden kaçınmak için)
    log_data = np.log1p(df[col])
    print(f"\nLog dönüşümü sonrası {col} normallik testi:")
    log_normal = dagostino_test(log_data)
    log_skewness = log_data.skew()
    
    # Kare kök dönüşümü
    sqrt_data = np.sqrt(df[col])
    print(f"\nKare kök dönüşümü sonrası {col} normallik testi:")
    sqrt_normal = dagostino_test(sqrt_data)
    sqrt_skewness = sqrt_data.skew()
    
    # BoxCox dönüşümü (sadece pozitif değerler için)
    try:
        boxcox_data, _ = boxcox(df[col])
        print(f"\nBoxCox dönüşümü sonrası {col} normallik testi:")
        boxcox_normal = dagostino_test(pd.Series(boxcox_data))
        boxcox_skewness = pd.Series(boxcox_data).skew()
    except:
        print("\nBoxCox dönüşümü uygulanamadı (negatif değerler olabilir)")
        boxcox_normal = False
        boxcox_skewness = None
    
    # Dönüşüm sonuçlarını tabloya ekle
    transformation_results.append({
        'Özellik': col,
        'Orijinal Normal mi?': original_normal,
        'Orijinal Skewness': original_skewness,
        'Log Normal mi?': log_normal,
        'Log Skewness': log_skewness,
        'Sqrt Normal mi?': sqrt_normal,
        'Sqrt Skewness': sqrt_skewness,
        'BoxCox Normal mi?': boxcox_normal,
        'BoxCox Skewness': boxcox_skewness
    })

# Dönüşüm sonuçları tablosunu oluştur
transformation_df = pd.DataFrame(transformation_results)
print("\nDönüşüm Karşılaştırma Tablosu:")
print(transformation_df.to_string(index=False))

# Dönüşüm karşılaştırmasını görselleştir (ilk 2 özellik için)
for col in numerical_cols[:2]:
    plt.figure(figsize=(15, 10))
    
    # Orijinal veri
    plt.subplot(2, 2, 1)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Orijinal {col} (Skewness: {df[col].skew():.2f})')
    
    # Log dönüşümü
    plt.subplot(2, 2, 2)
    sns.histplot(np.log1p(df[col].dropna()), kde=True)
    plt.title(f'Log Dönüşümü {col} (Skewness: {np.log1p(df[col]).skew():.2f})')
    
    # Sqrt dönüşümü
    plt.subplot(2, 2, 3)
    sns.histplot(np.sqrt(df[col].dropna()), kde=True)
    plt.title(f'Kare Kök Dönüşümü {col} (Skewness: {np.sqrt(df[col]).skew():.2f})')
    
    # BoxCox dönüşümü
    try:
        plt.subplot(2, 2, 4)
        boxcox_data, _ = boxcox(df[col].dropna())
        sns.histplot(boxcox_data, kde=True)
        plt.title(f'BoxCox Dönüşümü {col} (Skewness: {pd.Series(boxcox_data).skew():.2f})')
    except:
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, 'BoxCox dönüşümü uygulanamadı', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title(f'BoxCox Dönüşümü {col}')
    
    plt.tight_layout()
    plt.savefig(f'donusum_karsilastirma_{col}.png')
    print(f"\n{col} için dönüşüm karşılaştırması 'donusum_karsilastirma_{col}.png' olarak kaydedildi.")

print("\n7. NORMALİZASYON")
print("=" * 50)

# Sayısal sütunları normalize et
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("Normalizasyon sonrası veri seti başlığı:")
print(df.head().to_string())

print("\n8. SIMPLE LINEAR REGRESSION")
print("=" * 50)

# En yüksek korelasyona sahip özelliği seç
top_feature = correlation.index[1]  # İlk indeks CO2EMISSIONS'ın kendisi
print(f"En yüksek korelasyona sahip özellik: {top_feature}, korelasyon: {correlation[1]:.4f}")

# Veriyi eğitim ve test kümelerine ayır
X = df[[top_feature]]
y = df['CO2EMISSIONS']
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

print(f"Katsayı: {simple_model.coef_[0]:.4f}")
print(f"Kesişim: {simple_model.intercept_:.4f}")
print(f"Ortalama Kare Hata (MSE): {mse:.4f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.4f}")
print(f"R-kare (R²): {r2:.4f}")

# Çapraz doğrulama ile R² değerini hesapla
cv_scores = cross_val_score(simple_model, X, y, cv=5, scoring='r2')
print(f"Çapraz Doğrulama R²: {cv_scores.mean():.4f}")

# Basit doğrusal regresyonu görselleştir
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Gerçek Değerler')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Tahminler')
plt.title(f'Basit Doğrusal Regresyon: CO2EMISSIONS vs {top_feature}')
plt.xlabel(top_feature)
plt.ylabel('CO2EMISSIONS')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('basit_regresyon.png')
print("\nBasit doğrusal regresyon grafiği 'basit_regresyon.png' olarak kaydedildi.")

print("\n9. MULTIPLE LINEAR REGRESSION")
print("=" * 50)

# En yüksek korelasyona sahip ilk 3 özelliği seç
top_features = correlation.index[1:4]  # İlk indeks CO2EMISSIONS'ın kendisi
print(f"En yüksek korelasyona sahip 3 özellik: {', '.join(top_features)}")
print(f"Korelasyonlar: {correlation[1]:.4f}, {correlation[2]:.4f}, {correlation[3]:.4f}")

# Veriyi eğitim ve test kümelerine ayır
X_multi = df[top_features]
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

print("Katsayılar:")
for i, feature in enumerate(top_features):
    print(f"  {feature}: {multi_model.coef_[i]:.4f}")
print(f"Kesişim: {multi_model.intercept_:.4f}")
print(f"Ortalama Kare Hata (MSE): {mse_multi:.4f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse_multi:.4f}")
print(f"R-kare (R²): {r2_multi:.4f}")

# Çapraz doğrulama ile R² değerini hesapla
cv_scores_multi = cross_val_score(multi_model, X_multi, y, cv=5, scoring='r2')
print(f"Çapraz Doğrulama R²: {cv_scores_multi.mean():.4f}")

# İki modeli karşılaştır
print("\n10. MODEL KARŞILAŞTIRMASI")
print("=" * 50)
print(f"Basit Doğrusal Regresyon R²: {r2:.4f}")
print(f"Çoklu Doğrusal Regresyon R²: {r2_multi:.4f}")
improvement = ((r2_multi - r2) / r2) * 100
print(f"İyileştirme: {improvement:.2f}%")

# Tahmin örneği
print("\n11. MANUEL TAHMİN ÖRNEĞİ")
print("=" * 50)
# Örnek değerler
sample_values = {}
for feature in top_features:
    if feature == 'ENGINESIZE':
        sample_values[feature] = 2.5
    elif feature == 'FUELCONSUMPTION_COMB':
        sample_values[feature] = 10.5
    else:
        sample_values[feature] = df[feature].mean()

print("Örnek değerler:")
for feature, value in sample_values.items():
    print(f"  {feature}: {value:.2f}")

# Manuel tahmin
prediction = multi_model.intercept_
for i, feature in enumerate(top_features):
    prediction += multi_model.coef_[i] * sample_values[feature]

print(f"Tahmin edilen CO2 emisyonu: {math.floor(prediction)}")

# Sonuç
print("\n12. SONUÇ")
print("=" * 50)
print("Bu çalışmada şunları gerçekleştirdik:")
print("1. Veri setinin incelenmesi ve eksik değerlerin tespiti")
print("2. Eksik değerlerin doldurulması")
print("3. Kategorik değerlerin sayısal değerlere dönüştürülmesi (One-Hot Encoding)")
print("4. Korelasyon analizi")
print("5. D'Agostino K^2 testi ile normallik kontrolü")
print("6. Farklı dönüşüm tekniklerinin karşılaştırılması (BoxCox, Log, Square Root)")
print("7. Verilerin normalizasyonu (StandardScaler)")
print("8. Basit doğrusal regresyon modelinin oluşturulması ve değerlendirilmesi")
print("9. Çoklu doğrusal regresyon modelinin oluşturulması ve değerlendirilmesi")
print("10. Modellerin karşılaştırılması")
print("11. Manuel tahmin örneği")