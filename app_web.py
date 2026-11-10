"""
╔════════════════════════════════════════════════════════════════════════════╗
║   DetectaVial Ecuador — Edición Ultra Elegante (Apple iOS Minimalist)      ║
║   Visión Artificial en Tiempo Real con Cálculo de Distancia en Metros      ║
║   Normativa MTOP / INEN (Ecuador)                                          ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import io
import time
import csv
import datetime
from pathlib import Path
from collections import Counter
from PIL import Image

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

from config_senales import (
    TRADUCTOR,
    COLOR_MAP,
    EMOJI_MAP,
    CORRECCIONES_OVERRIDE,
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

st.set_page_config(
    page_title="DetectaVial Ecuador · Visor Pro",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stMarkdown {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0f172a;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(238, 245, 255, 0.95) 0%, rgba(255, 255, 255, 1) 55%),
                radial-gradient(circle at 85% 85%, rgba(243, 240, 255, 0.75) 0%, rgba(255, 255, 255, 1) 60%);
    background-attachment: fixed;
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.82) !important;
    backdrop-filter: blur(30px) saturate(180%) !important;
    border-right: 1px solid rgba(226, 232, 240, 0.85) !important;
}

.hero-wrapper { text-align: center; padding: 2.2rem 1rem 1.4rem; }
.hero-title {
    font-size: 2.9rem; font-weight: 800; letter-spacing: -1.2px; line-height: 1.15;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0071e3 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;
}

.glass-card {
    background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(24px) saturate(180%);
    border-radius: 26px; padding: 1.8rem 2rem; border: 1px solid rgba(255, 255, 255, 0.95);
    box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.05); margin-bottom: 1.6rem;
}

.camera-box-container {
    background: linear-gradient(145deg, #090d16 0%, #0f172a 100%);
    border-radius: 28px; padding: 1.2rem; border: 2px solid rgba(0, 113, 227, 0.35);
    box-shadow: 0 25px 60px -15px rgba(15, 23, 42, 0.35); margin-bottom: 1.5rem;
}

.verify-card {
    background: rgba(255, 255, 255, 0.95); border-radius: 22px; padding: 1.4rem 1.6rem;
    border: 1px solid rgba(226, 232, 240, 0.9); box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04);
    margin-top: 1.2rem; display: flex; align-items: center; justify-content: space-between;
}
.verify-badge-verified {
    background: rgba(34, 197, 94, 0.1); color: #16a34a; border: 1px solid rgba(34, 197, 94, 0.25);
    padding: 0.38rem 0.95rem; border-radius: 12px; font-size: 0.82rem; font-weight: 800;
}

.stButton > button {
    background: linear-gradient(135deg, #0071e3 0%, #005bb5 100%) !important; color: #ffffff !important;
    border: none !important; border-radius: 16px !important; padding: 0.72rem 1.8rem !important;
    font-weight: 600 !important; box-shadow: 0 8px 20px rgba(0, 113, 227, 0.25) !important;
}
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(226, 232, 240, 0.9), transparent); margin: 1.8rem 0; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "best.pt"
ALTURA_REAL_SENAL_M = 0.60
FOCAL_LENGTH_ESTIMADO = 600

@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta: str) -> "YOLO | None":
    if not YOLO_AVAILABLE or not Path(ruta).exists(): return None
    return YOLO(ruta)

def estimar_distancia(altura_pixel: int) -> float:
    if altura_pixel <= 0: return 0.0
    return (ALTURA_REAL_SENAL_M * FOCAL_LENGTH_ESTIMADO) / altura_pixel

def registrar_senal_desconocida(clase_orig: str):
    archivo_registro = "senales_por_agregar.csv"
    if "señales_registradas" not in st.session_state:
        st.session_state.señales_registradas = set()
        
    if clase_orig not in st.session_state.señales_registradas:
        st.session_state.señales_registradas.add(clase_orig)
        existe = os.path.isfile(archivo_registro)
        with open(archivo_registro, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not existe:
                writer.writerow(["Fecha_Hora", "ID_o_Nombre_Original", "Sugerencia_Accion"])
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                clase_orig, 
                "Agregar a config_senales.py"
            ])

def traducir_clase(nombre_clase: str | int, model_names: dict | None = None) -> str:
    global modelo
    if model_names is None:
        try:
            if "modelo" in globals() and modelo is not None and hasattr(modelo, "names"):
                model_names = modelo.names
        except Exception: model_names = None

    orig_str = str(nombre_clase).strip()
    orig_lower = orig_str.lower()

    if orig_str in CORRECCIONES_OVERRIDE: return CORRECCIONES_OVERRIDE[orig_str]
    for k, v in CORRECCIONES_OVERRIDE.items():
        if k.lower() == orig_lower: return v

    if orig_str in TRADUCTOR: return TRADUCTOR[orig_str]
    for k, v in TRADUCTOR.items():
        if k.lower() == orig_lower: return v

    nombre_resuelto = None
    if model_names:
        if orig_str.lstrip("-").isdigit() and int(orig_str) in model_names:
            nombre_resuelto = str(model_names[int(orig_str)])
        elif isinstance(nombre_clase, int) and nombre_clase in model_names:
            nombre_resuelto = str(model_names[nombre_clase])

    if nombre_resuelto:
        res_lower = nombre_resuelto.lower()
        if nombre_resuelto in CORRECCIONES_OVERRIDE: return CORRECCIONES_OVERRIDE[nombre_resuelto]
        for k, v in TRADUCTOR.items():
            if k.lower() == res_lower: return v
        return f"⚠️ Nueva Señal: {nombre_resuelto}"
    return f"⚠️ Nueva Señal: {orig_str}"

def get_color(nombre_es: str) -> tuple:
    if nombre_es in COLOR_MAP: return COLOR_MAP[nombre_es]
    return (200, 0, 255)

def clasificar_tipo_senal(nombre_es: str) -> dict:
    if "Nueva Señal" in nombre_es or nombre_es.startswith("⚠️"):
        return {"categoria": "Señal No Mapeada", "tag": "ALERTA", "color": "#c026d3", "bg": "rgba(192, 38, 211, 0.08)", "valido": False}
    if any(k.lower() in nombre_es.lower() for k in ["pare", "ceda", "no ", "prohibido", "límite", "girar", "estacionar", "vía", "velocidad"]):
        return {"categoria": "Reglamentaria (Serie R)", "tag": "OBLIGATORIA", "color": "#dc2626", "bg": "rgba(220, 38, 38, 0.08)", "valido": True}
    if any(k.lower() in nombre_es.lower() for k in ["curva", "peligro", "reductor", "escolar", "peaton", "derrumbe", "semáforo"]):
        return {"categoria": "Preventiva (Serie P)", "tag": "PREVENCIÓN", "color": "#d97706", "bg": "rgba(217, 119, 6, 0.08)", "valido": True}
    return {"categoria": "Informativa (Serie I)", "tag": "INFORMACIÓN", "color": "#2563eb", "bg": "rgba(37, 99, 235, 0.08)", "valido": True}

def dibujar_detecciones(imagen_bgr: np.ndarray, results) -> tuple[np.ndarray, list[dict]]:
    detecciones = []
    img = imagen_bgr.copy()

    for result in results:
        boxes = result.boxes
        nombres = result.names
        if boxes is None or len(boxes) == 0: continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            nombre_orig = nombres.get(cls_id, str(cls_id)) if nombres else str(cls_id)
            nombre_es = traducir_clase(nombre_orig, model_names=nombres)
            
            if "Nueva Señal" in nombre_es:
                registrar_senal_desconocida(nombre_orig)

            color_bgr = get_color(nombre_es)
            altura_px = y2 - y1
            distancia_m = estimar_distancia(altura_px)

            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
            
            label_text = f"{nombre_es} | {distancia_m:.1f}m | {conf:.0%}"
            font_scale = 0.58
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thick)

            lx1, ly1 = x1, max(y1 - th - 12, 0)
            lx2, ly2 = x1 + tw + 12, max(y1, th + 12)

            overlay = img.copy()
            cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color_bgr, -1)
            cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)
            cv2.putText(img, label_text, (lx1 + 6, ly2 - 6), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

            detecciones.append({
                "clase_original": nombre_orig, "clase_es": nombre_es, 
                "confianza": conf, "distancia": distancia_m, "bbox": (x1, y1, x2, y2)
            })
    return img, detecciones

st.markdown("""
<div class="hero-wrapper">
    <div class="hero-title">DetectaVial ADAS</div>
    <p class="hero-subtitle">Visor inteligente con cálculo de distancia en tiempo real (Normativa MTOP / INEN)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="font-weight:800; font-size:1.15rem; margin-bottom:1rem;">🎛️ Modos de Inspección</div>', unsafe_allow_html=True)
    modo = st.radio("Entrada", ["📹 Cámara en Vivo (Local)", "📸 Foto desde Celular/PC", "🖼️ Subir Imagen"], label_visibility="collapsed")
    confianza = st.slider("Sensibilidad", 0.10, 0.95, 0.35, 0.05)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if Path(MODEL_PATH).exists():
        st.markdown("""
        <div style="background: rgba(255,255,255,0.7); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 14px 18px; display: flex; align-items: center; gap: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.02);">
            <div style="position: relative; width: 12px; height: 12px;">
                <div style="position: absolute; width: 100%; height: 100%; border-radius: 50%; background: #10b981; animation: pulseGlow 2s infinite;"></div>
                <div style="position: absolute; width: 100%; height: 100%; border-radius: 50%; background: #10b981;"></div>
            </div>
            <div style="font-weight: 700; color: #065f46; font-size: 0.95rem; letter-spacing: -0.2px;">IA Lista y Activa</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"❌ No se encontró '{MODEL_PATH}'")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:800; font-size:1.15rem; margin-bottom:1rem;">📥 Exportar Datos</div>', unsafe_allow_html=True)
    
    if os.path.exists("senales_por_agregar.csv"):
        with open("senales_por_agregar.csv", "rb") as f:
            st.download_button(
                label="Descargar Señales Nuevas (CSV)",
                data=f,
                file_name="senales_nuevas.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Aún no hay señales nuevas.")

modelo = cargar_modelo(MODEL_PATH)

if "📹" in modo:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📹 Radar en Vivo Continúo")
    st.caption("Transmisión en tiempo real. La IA dibujará los cuadros y la distancia directamente sobre el video.")
    st.markdown('</div>', unsafe_allow_html=True)

    def procesar_frame(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if modelo is not None:
            results = modelo.predict(source=img, conf=confianza, verbose=False)
            img_anotada, _ = dibujar_detecciones(img, results)
        else:
            img_anotada = img
        return av.VideoFrame.from_ndarray(img_anotada, format="bgr24")

    rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [
                    "turn:openrelay.metered.ca:80",
                    "turn:openrelay.metered.ca:443"
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    })

    st.markdown('<div class="camera-box-container">', unsafe_allow_html=True)
    webrtc_streamer(
        key="radar-vivo",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_frame_callback=procesar_frame,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

elif "📸" in modo or "🖼️" in modo:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if "📸" in modo:
        st.markdown("### 📸 Escáner Móvil")
        archivo = st.camera_input("Capturar señal")
    else:
        st.markdown("### 🖼️ Analizar Imagen")
        archivo = st.file_uploader("Subir foto", type=["jpg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo:
        img_np = np.array(Image.open(archivo).convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = modelo.predict(source=img_bgr, conf=confianza, verbose=False)
        img_anotada, dets = dibujar_detecciones(img_bgr, results)
        
        st.image(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if dets:
            top = max(dets, key=lambda x: x["confianza"])
            nombre, conf, dist = top["clase_es"], top["confianza"], top["distancia"]
            info = clasificar_tipo_senal(nombre)
            emoji = EMOJI_MAP.get(nombre, "⚠️" if not info["valido"] else "🔷")

            st.markdown(f"""
            <div class="verify-card" style="border-left: 6px solid {info['color']};">
                <div style="display:flex; align-items:center; gap:1.1rem;">
                    <div style="font-size:2.4rem; background:{info['bg']}; width:62px; height:62px; border-radius:18px; display:flex; align-items:center; justify-content:center;">
                        {emoji}
                    </div>
                    <div>
                        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
                            <span style="font-size:1.3rem; font-weight:800; color:#0f172a;">{nombre}</span>
                            <span style="background:{info['bg']}; color:{info['color']}; font-size:0.75rem; font-weight:800; padding:0.2rem 0.65rem; border-radius:8px;">{info['tag']}</span>
                        </div>
                        <div style="font-size:0.86rem; color:#64748b; margin-bottom:0.3rem;">
                            {info['categoria']} · Certeza: <strong style="color:#0f172a;">{conf:.1%}</strong> · 📏 Distancia: <strong style="color:#0071e3; font-size:1rem;">{dist:.1f} m</strong>
                        </div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <span class="{'verify-badge-verified' if info['valido'] else 'verify-badge-alert'}">
                        {'✓ VERIFICADA' if info['valido'] else '⚠️ NO MAPEADA'}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="verify-card" style="background: rgba(248, 250, 252, 0.9);">
                <div style="display:flex; align-items:center; gap:0.9rem;">
                    <span style="font-size:1.9rem;">🔎</span>
                    <div>
                        <div style="font-weight:700; color:#334155; font-size:1rem;">Sin detecciones</div>
                        <div style="font-size:0.84rem; color:#64748b;">No se detectaron señales en esta imagen.</div>
                    </div>
                </div>
                <span style="font-size:0.8rem; color:#94a3b8; font-weight:700;">VACÍO</span>
            </div>
            """, unsafe_allow_html=True)
