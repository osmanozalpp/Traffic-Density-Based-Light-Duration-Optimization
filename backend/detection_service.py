from ultralytics import YOLO
import numpy as np
import torch

class DetectionService:
    def __init__(self, model_path="models/yolov8n.pt", device=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        YOLOv8 modelini yükler. Belirli araç sınıflarını algılamak üzere yapılandırılmıştır.

        Args:
            model_path (str): YOLOv8 model yolu
            device (str): 'cuda' veya 'cpu'. Belirtilmezse otomatik algılanır
            conf_threshold (float): Minimum güven skoru (0-1)
            iou_threshold (float): NMS için IOU eşiği
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.allowed_classes = {'car', 'bus', 'motorcycle', 'truck'}
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray):
        """
        Bir karedeki araçları tespit eder.

        Args:
            frame (np.ndarray): OpenCV görüntü karesi

        Returns:
            List[Dict]: [
                {
                    "type": "car" / "bus" / "motorcycle",
                    "bbox": (x1, y1, x2, y2)
                },
                ...
            ]
        """
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls.item())
            label = self.model.names[class_id]

            if label in self.allowed_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "type": label,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections
