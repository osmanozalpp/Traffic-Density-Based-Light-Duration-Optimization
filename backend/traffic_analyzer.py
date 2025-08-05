import numpy as np

class TrafficAnalyzer:
    def __init__(self, history_limit=30):
        self.history = []
        self.history_limit = history_limit

    def _get_type_weight(self, vehicle_type):
        weights = {
            'motorcycle': 0.3,
            'car': 1.0,
            'bus': 1.8,
            'truck': 2.0
        }
        return weights.get(vehicle_type, 1.0)

    def calculate_density(self, tracked_detections, frame_height):
        """
        Gelişmiş trafik yoğunluğu hesaplama:
        - Araç tipi (ağırlık)
        - Görüntüde kapladığı alan (normalize)
        - Alt kısma yakınlık (ön sırada olma)
        - Bekleme süresi (hareketsiz kaldığı frame sayısı)
        """
        score = 0

        for d in tracked_detections:
            x1, y1, x2, y2 = d["bbox"]
            area = (x2 - x1) * (y2 - y1)
            normalized_area = min(area / 15000, 2.0)  # çok büyük alanları sınırlıyoruz
            vertical_position = y2 / frame_height  # alt kısımlara yaklaşma
            still_frames = d.get("frame_still_count", 0)
            wait_bonus = min(still_frames / 30, 2.0)  # 1 saniye = +1 katkı, max 2

            type_weight = self._get_type_weight(d["type"])

            # Logaritmik katkı: alt kısma yakınlık çok agresif etkimesin
            position_bonus = 0.5 + np.log1p(vertical_position)

            vehicle_score = (
                type_weight *
                normalized_area *
                position_bonus *
                (1.0 + wait_bonus)
            )
            score += vehicle_score

        # Ortalama alıyoruz (moving average gibi)
        self.history.append(score)
        if len(self.history) > self.history_limit:
            self.history.pop(0)

        return round(np.mean(self.history), 2)

    def suggest_green_time(self, density_score):
        """
        Trafik yoğunluğuna göre yeşil ışık süresi önerisi (saniye).
        Daha yumuşak geçişler içeriyor.
        """
        if density_score < 3:
            return 15
        elif density_score < 5:
            return 20
        elif density_score < 7:
            return 25
        elif density_score < 9:
            return 30
        elif density_score < 12:
            return 35
        elif density_score < 16:
            return 40
        elif density_score < 20:
            return 45
        else:
            return 60
