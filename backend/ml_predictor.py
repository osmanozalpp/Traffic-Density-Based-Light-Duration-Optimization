import joblib
import pandas as pd

class GreenLightPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict_duration(self, vehicle_count, car_ratio, bus_ratio, motorcycle_ratio, density_score):
        features = pd.DataFrame([{
            "vehicle_count": vehicle_count,
            "car_ratio": car_ratio,
            "bus_ratio": bus_ratio,
            "motorcycle_ratio": motorcycle_ratio,
            "density_score": density_score
        }])
        return int(self.model.predict(features)[0])
