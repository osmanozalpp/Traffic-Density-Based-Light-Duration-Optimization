from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Encode edilecek değerler
weather_conditions = ["Clear", "Cloudy", "Rainy", "Snowy"]

# LabelEncoder nesnesi oluştur ve eğit
encoder = LabelEncoder()
encoder.fit(weather_conditions)

# frontend/models klasörüne kaydet
model_dir = os.path.join("frontend", "models")
os.makedirs(model_dir, exist_ok=True)

encoder_path = os.path.join(model_dir, "weather_encoder.pkl")
joblib.dump(encoder, encoder_path)

print(f"✅ Encoder başarıyla '{encoder_path}' olarak kaydedildi.")
