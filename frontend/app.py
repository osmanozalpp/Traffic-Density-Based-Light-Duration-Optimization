import sys
import os
import streamlit as st
import cv2
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

green_light_model_path = os.path.join(BASE_DIR, "frontend", "models", "green_light_predictor.pkl")
yolo_model_path = os.path.join(BASE_DIR, "frontend", "models", "yolov8n.pt")
encoder_path = os.path.join(BASE_DIR, "frontend", "models", "weather_encoder.pkl")

from backend.detection_service import DetectionService
from backend.traffic_analyzer import TrafficAnalyzer
from backend.simple_tracker import SimpleTracker
from backend.ml_predictor import GreenLightPredictor

st.set_page_config(page_title="🚦 Trafik Optimizasyonu", layout="wide")

style_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(style_path, "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Arayüz başlığı
st.markdown('''
<div class="navbar">
    <span class="nav-title">🚦 Trafik Yoğunluğuna Göre Işık Süresi Optimizasyonu</span>
</div>
<div class="flow-container">
    <div class="step-box">📤<br>Video Yükleme</div>
    <div class="step-box">🧠<br>Araçları Tespit Etme</div>
    <div class="step-box">⏱️<br>Yeşil Işık Süresini Tahmin Etme</div>
    <div class="step-box">♻️<br>Sürekli Güncelleme</div>
</div>
<div class="video-container">
    <p>Bu proje, kavşaklardaki trafik yoğunluğunu gerçek zamanlı analiz ederek trafik ışığı sürelerini dinamik olarak optimize eden bir sistemdir. Sistem ilk olarak yüklenen trafik videosunu kare kare işler ve YOLOv8 nesne algılama modeli ile araçları tespit eder. Tespit edilen araçlar, sınıflarına göre (otomobil, otobüs, motosiklet vb.) ayrılır ve konumlarına göre yoğunluk analizi yapılır. Ardından, yoğunluk verileri ve araç sayıları bir makine öğrenmesi modeli (Random Forest Regressor) ile değerlendirilerek ideal yeşil ışık süresi tahmin edilir. </p>
</div>
''', unsafe_allow_html=True)

# Video yükleme alanı
st.markdown("### 📤 Trafik videosunu yükleyin")
uploaded_file = st.file_uploader("Bir trafik videosu seçin", type=["mp4", "avi", "mov"])

if uploaded_file and st.button("🚦 Analizi Başlat"):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    detector = DetectionService(yolo_model_path)
    analyzer = TrafficAnalyzer()
    tracker = SimpleTracker()
    predictor = GreenLightPredictor(green_light_model_path, encoder_path)

    predicted_green_history = []
    vehicle_count_history = []
    frame_index = 0

    # Üst bölüm (video ve istatistikler)
    st.markdown('<div class="analyze-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        stframe = st.empty()
    with col2:
        result_box = st.empty()
        stats_box = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # Alt bölüm: grafik kutusu
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    line_chart_area = g1.empty()
    bar_chart_area = g2.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        count = len(tracked)
        height = frame.shape[0]
        density = analyzer.calculate_density(tracked, height)

        types = {"car": 0, "bus": 0, "motorcycle": 0}
        for d in tracked:
            if d["type"] in types:
                types[d["type"]] += 1

        hour = datetime.now().hour
        predicted = predictor.predict_duration(
            vehicle_count=count,
            car_ratio=types["car"] / (count or 1),
            bus_ratio=types["bus"] / (count or 1),
            motorcycle_ratio=types["motorcycle"] / (count or 1),
            density_score=density,
            time_of_day=hour,
            weather_condition="Clear"
        )

        predicted_green_history.append(predicted)
        vehicle_count_history.append(count)

        for d in tracked:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['type']}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        stframe.image(frame, channels="BGR", use_container_width=True)

        trafik_durumu = "Az Yoğun" if count < 4 else "Yoğun"
        result_box.markdown(f'''
        <div class="info-box">
            <h3>📊 Anlık Analiz Sonuçları</h3>
            <b>Araç Sayısı:</b> <span style="color:skyblue;font-size:18px;">{count}</span><br>
            <b>Trafik Durumu:</b> <span class="highlight-traffic">{trafik_durumu}</span><br>
            <b>Yeşil Işık Süresi:</b> <span style="color:lime;font-size:18px;"><b>{int(predicted)}s</b></span><br>
            <b>Trafik Yoğunluğu:</b> {round(density, 1)}<br>
            <b>Zaman:</b> {round(frame_index / cap.get(cv2.CAP_PROP_FPS), 1)}s
        </div>
        ''', unsafe_allow_html=True)

        ort_arac = round(np.mean(vehicle_count_history), 1)
        maks_arac = np.max(vehicle_count_history)
        ort_yesil = round(np.mean(predicted_green_history), 1)

        stats_box.markdown(f'''
        <div class="info-box">
            <h3>🗒️ Genel İstatistikler</h3>
            <b>İşlenen Frame:</b> {frame_index}<br>
            <b>Ortalama Araç:</b> {ort_arac}<br>
            <b>Maksimum Araç:</b> {maks_arac}<br>
            <b>Ort. Yeşil Işık:</b> {ort_yesil}s
        </div>
        ''', unsafe_allow_html=True)

        # Grafik: Araç Sayısı
        fig1, ax1 = plt.subplots(figsize=(5, 3), facecolor='black')
        ax1.plot(vehicle_count_history, color='deepskyblue')
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        ax1.set_title("Araç Sayısı", color='white')
        ax1.set_yticks(np.arange(min(vehicle_count_history), max(vehicle_count_history)+1, 1))
        fig1.patch.set_alpha(0.0)
        line_chart_area.pyplot(fig1)

        # Grafik: Yeşil Işık Süresi
        fig2, ax2 = plt.subplots(figsize=(5, 3), facecolor='black')
        ax2.bar(range(len(predicted_green_history)), predicted_green_history, color='lime')
        ax2.set_facecolor('black')
        ax2.tick_params(colors='white')
        ax2.set_title("Yeşil Işık Süresi", color='white')
        fig2.patch.set_alpha(0.0)
        bar_chart_area.pyplot(fig2)

        frame_index += 1

    cap.release()
