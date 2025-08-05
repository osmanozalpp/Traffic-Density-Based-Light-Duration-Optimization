import joblib
import pandas as pd

class GreenLightPredictor:
    def __init__(self, model_path, encoder_path):
        self.model = joblib.load(model_path)
        self.weather_encoder = joblib.load(encoder_path)

    def predict_duration(self, vehicle_count, car_ratio, bus_ratio, motorcycle_ratio, density_score, time_of_day, weather_condition):
        features = pd.DataFrame([{
            "vehicle_count": vehicle_count,
            "car_ratio": car_ratio,
            "bus_ratio": bus_ratio,
            "motorcycle_ratio": motorcycle_ratio,
            "density_score": density_score,
            "time_of_day": time_of_day,
            "weather_condition": self.weather_encoder.transform([weather_condition])[0]
        }])
        return int(self.model.predict(features)[0])
