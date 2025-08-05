# train_model.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Veri kümesini yükle
df = pd.read_csv("green_light_dataset.csv")

# Hava durumu verisini sayısal değere çevir (Label Encoding)
weather_encoder = LabelEncoder()
df["weather_condition"] = weather_encoder.fit_transform(df["weather_condition"])

# Özellikleri ve hedef değişkeni ayır
X = df.drop("green_light_duration", axis=1)
y = df["green_light_duration"]

# Eğitim/test seti ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Hata ölçümü
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"✅ MAE: {mae:.2f} saniye")

# Model ve encoder dosyasını kaydet
model_dir = os.path.join("frontend", "models")  # frontend/models içine kayıt
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "green_light_predictor.pkl"))
joblib.dump(weather_encoder, os.path.join(model_dir, "weather_encoder.pkl"))

print(f"✅ Model ve encoder başarıyla '{model_dir}' klasörüne kaydedildi.")
