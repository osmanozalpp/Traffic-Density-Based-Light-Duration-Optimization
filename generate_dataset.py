import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Sentetik veri üretimi
rows = 500
data = {
    "vehicle_count": np.random.randint(0, 100, size=rows),
    "car_ratio": np.round(np.random.uniform(0.3, 0.7, size=rows), 2),
    "bus_ratio": np.round(np.random.uniform(0.0, 0.3, size=rows), 2),
    "motorcycle_ratio": np.round(np.random.uniform(0.0, 0.3, size=rows), 2),
    "density_score": np.round(np.random.uniform(10, 300, size=rows), 2),
    "time_of_day": np.random.randint(0, 24, size=rows),  # 0–23 saat
    "weather_condition": np.random.choice(["Clear", "Rain", "Snow", "Fog", "Cloudy"], size=rows)
}

df = pd.DataFrame(data)

# Gerçekçi yeşil ışık süreleri üret
def calculate_duration(row):
    base = 30
    base += int(row["vehicle_count"] * 0.5)
    base += int(row["density_score"] * 0.1)
    if row["weather_condition"] in ["Rain", "Snow", "Fog"]:
        base += 10
    if row["time_of_day"] in range(7, 10) or row["time_of_day"] in range(16, 19):  # Yoğun saat
        base += 15
    return max(15, min(base, 120))

df["green_light_duration"] = df.apply(calculate_duration, axis=1)

# CSV olarak kaydet
df.to_csv("green_light_dataset.csv", index=False)
print("✅ green_light_dataset.csv başarıyla oluşturuldu.")
