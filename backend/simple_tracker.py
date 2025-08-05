import math

class SimpleTracker:
    def __init__(self, max_distance=50, max_miss_frames=10):
        self.next_id = 0
        self.vehicles = {}  # id: {bbox, still_count, miss_frames}
        self.max_distance = max_distance
        self.max_miss_frames = max_miss_frames

    def update(self, detections):
        updated_detections = []
        matched_ids = set()

        # Tüm araçları potansiyel olarak eşleşmemiş kabul et
        for vid in self.vehicles:
            self.vehicles[vid]['miss_frames'] += 1

        for det in detections:
            matched_id = None
            cx, cy = self._get_center(det["bbox"])

            for vid, vdata in self.vehicles.items():
                if vid in matched_ids:
                    continue
                pcx, pcy = self._get_center(vdata["bbox"])
                dist = self._distance_between((cx, cy), (pcx, pcy))

                if dist < self.max_distance:
                    matched_id = vid
                    break

            if matched_id is not None:
                prev_bbox = self.vehicles[matched_id]['bbox']
                moved = prev_bbox != det["bbox"]

                self.vehicles[matched_id]['bbox'] = det["bbox"]
                self.vehicles[matched_id]['still_count'] = 1 if moved else self.vehicles[matched_id]['still_count'] + 1
                self.vehicles[matched_id]['miss_frames'] = 0
            else:
                matched_id = self.next_id
                self.vehicles[matched_id] = {
                    'bbox': det["bbox"],
                    'still_count': 1,
                    'miss_frames': 0
                }
                self.next_id += 1

            matched_ids.add(matched_id)

            updated_detections.append({
                "id": matched_id,
                "type": det["type"],
                "bbox": det["bbox"],
                "frame_still_count": self.vehicles[matched_id]['still_count']
            })

        # Eşleşmeyenleri sil (örnek: araç çıkmış olabilir)
        self.vehicles = {vid: data for vid, data in self.vehicles.items() if data['miss_frames'] <= self.max_miss_frames}

        return updated_detections

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _distance_between(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
