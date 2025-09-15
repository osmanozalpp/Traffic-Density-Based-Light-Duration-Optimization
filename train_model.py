# train_model.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# ✅ Güncel veri setini yükle
df = pd.read_csv("green_light_dataset.csv")

# 🎯 Özellikler ve hedef değişken
X = df.drop("green_light_duration", axis=1)
y = df["green_light_duration"]

# 🔀 Eğitim/test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌲 Modeli eğit
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📏 Başarı ölçümü
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"✅ Ortalama mutlak hata (MAE): {mae:.2f} saniye")

# 💾 Modeli kaydet
model_dir = os.path.join("frontend", "models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "green_light_predictor.pkl"))
print(f"✅ Model başarıyla '{model_dir}/green_light_predictor.pkl' dosyasına kaydedildi.")
