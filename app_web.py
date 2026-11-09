"""
╔══════════════════════════════════════════════════════════════════╗
║   DetectaVial Ecuador — Aplicación Web de Señales de Tránsito   ║
║   Desarrollada con Streamlit + Ultralytics YOLOv8               ║
║   Normativa MTOP / INEN (Ecuador)                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import io
import time
from pathlib import Path
from collections import Counter
from PIL import Image

# ── Importación de Configuración Modular y Diccionarios ───────────────────────
from config_senales import (
    TRADUCTOR,
    COLOR_MAP,
    EMOJI_MAP,
    CORRECCIONES_OVERRIDE,
)

# ── Importación de Ultralytics YOLO ──────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE LA PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DetectaVial Ecuador",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  ESTILOS CSS PERSONALIZADOS (ESTILO iOS Y MINIMALISTA)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Tipografía limpia estilo San Francisco / Roboto ── */
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Roboto',
                 'Helvetica Neue', sans-serif;
    color: #1c1c1e;
}

/* ── Fondo minimalista en degradé suave ── */
.main {
    background: linear-gradient(135deg, #f5f5f7 0%, #fafafa 50%, #eff3f8 100%);
    min-height: 100vh;
}

/* ── Barra lateral (Sidebar iOS Glassmorphism) ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.82) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border-right: 1px solid rgba(0, 0, 0, 0.06) !important;
    box-shadow: 2px 0 20px rgba(0, 0, 0, 0.03) !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

/* ── Encabezado Principal (Hero) ── */
.hero-title {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
}

.hero-title h1 {
    font-size: 2.7rem;
    font-weight: 700;
    letter-spacing: -0.6px;
    background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #2563eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    color: #6b7280;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1.6rem;
    text-align: center;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(37, 99, 235, 0.08);
    color: #2563eb;
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    font-size: 0.83rem;
    font-weight: 600;
    border: 1px solid rgba(37, 99, 235, 0.18);
}

/* ── Tarjetas Minimalistas (Cards iOS) ── */
.card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
    border: 1px solid rgba(0, 0, 0, 0.05);
    margin-bottom: 1.4rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.07);
}

/* ── Botones iOS con Bordes Redondeados Amplios y Sombras ── */
.stButton > button {
    background: linear-gradient(135deg, #0071e3 0%, #0056b3 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.7rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: -0.2px !important;
    box-shadow: 0 4px 14px rgba(0, 113, 227, 0.28) !important;
    transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(0, 113, 227, 0.38) !important;
    background: linear-gradient(135deg, #0a84ff 0%, #0066cc 100%) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(0, 113, 227, 0.2) !important;
}

/* ── Botones de Descarga ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #34c759 0%, #28a745 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 4px 14px rgba(52, 199, 89, 0.25) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(52, 199, 89, 0.35) !important;
}

/* ── Componente de Subida de Archivos ── */
[data-testid="stFileUploader"] {
    background: rgba(245, 245, 247, 0.8) !important;
    border: 2px dashed rgba(0, 113, 227, 0.3) !important;
    border-radius: 18px !important;
    padding: 1.2rem !important;
    transition: all 0.25s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 113, 227, 0.65) !important;
    background: rgba(0, 113, 227, 0.03) !important;
}

/* ── Radio Buttons en Sidebar ── */
[data-testid="stRadio"] > div {
    gap: 0.5rem;
}

[data-testid="stRadio"] > div > label {
    background: rgba(255, 255, 255, 0.9);
    border: 1.5px solid rgba(0, 0, 0, 0.06);
    border-radius: 14px;
    padding: 0.75rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    font-weight: 500;
    color: #374151;
    transition: all 0.2s ease;
}

[data-testid="stRadio"] > div > label:hover {
    border-color: #0071e3;
    color: #0071e3;
    background: rgba(0, 113, 227, 0.04);
}

/* ── Métricas y Estadísticas ── */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
}

[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #0071e3 !important;
}

[data-testid="stMetricLabel"] {
    font-weight: 500 !important;
    color: #6b7280 !important;
}

/* ── Barra de Progreso iOS ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #0071e3 0%, #5ac8fa 100%) !important;
    border-radius: 10px !important;
    height: 10px !important;
}

/* ── Tarjeta de Señal Detectada ── */
.signal-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.65rem;
    border: 1px solid rgba(0, 0, 0, 0.06);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    transition: all 0.2s ease;
}

.signal-card:hover {
    border-color: rgba(0, 113, 227, 0.3);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    transform: translateX(3px);
}

.signal-name {
    font-weight: 600;
    color: #111827;
    font-size: 0.96rem;
}

.signal-badge {
    background: rgba(0, 113, 227, 0.1);
    color: #0071e3;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-weight: 700;
    font-size: 0.85rem;
}

.signal-conf {
    color: #6b7280;
    font-size: 0.82rem;
    margin-top: 2px;
}

/* ── Indicadores de Confianza ── */
.conf-high   { color: #28a745; font-weight: 600; }
.conf-medium { color: #f59e0b; font-weight: 600; }
.conf-low    { color: #ef4444; font-weight: 600; }

/* ── Divisores y Alertas ── */
.divider {
    height: 1px;
    background: rgba(0, 0, 0, 0.06);
    margin: 1.5rem 0;
}

.alert-success {
    background: rgba(52, 199, 89, 0.1);
    border: 1px solid rgba(52, 199, 89, 0.28);
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    color: #166534;
    font-weight: 500;
    font-size: 0.9rem;
}

.alert-info {
    background: rgba(0, 113, 227, 0.07);
    border: 1px solid rgba(0, 113, 227, 0.2);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    color: #1e40af;
    font-weight: 500;
}

/* Ocultar elementos nativos innecesarios de Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE DETECCIÓN Y LÓGICA DE TRADUCCIÓN JERÁRQUICA
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "best.pt"


@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta: str) -> "YOLO | None":
    """Carga el modelo YOLOv8 una sola vez y lo mantiene en memoria."""
    if not YOLO_AVAILABLE:
        return None
    if not Path(ruta).exists():
        return None
    return YOLO(ruta)


def traducir_clase(nombre_clase: str | int, model_names: dict | None = None) -> str:
    """
    Traduce el nombre o ID de clase aplicando una jerarquía de 3 niveles:
      1. Reglas de Corrección y Anulación (CORRECCIONES_OVERRIDE):
         Sobrescribe anomalías del dataset (ej: corregir falsos 'Semáforos' a 'Pare').
      2. Diccionario Traductor General (TRADUCTOR):
         Mapea IDs y nombres estándar al reglamento MTOP/INEN.
         Si es un ID numérico, consulta automáticamente 'model_names' (o 'modelo.names').
      3. Fallback de Alerta Fucsia:
         Si no existe en ninguno, devuelve: '⚠️ Nueva Señal: [ID_o_Nombre_Original]'
    """
    global modelo

    # Intentar obtener la lista interna del modelo si no fue provista explícitamente
    if model_names is None:
        try:
            if "modelo" in globals() and modelo is not None and hasattr(modelo, "names"):
                model_names = modelo.names
        except Exception:
            model_names = None

    orig_str = str(nombre_clase).strip()
    orig_lower = orig_str.lower()

    # ──────────────────────────────────────────────────────────────────────────
    # PASO 1: Reglas de Corrección / Anulación prioritarias (Overrides)
    # ──────────────────────────────────────────────────────────────────────────
    if orig_str in CORRECCIONES_OVERRIDE:
        return CORRECCIONES_OVERRIDE[orig_str]

    for k, v in CORRECCIONES_OVERRIDE.items():
        if k.lower() == orig_lower:
            return v

    # ──────────────────────────────────────────────────────────────────────────
    # PASO 2: Diccionario Traductor General
    # ──────────────────────────────────────────────────────────────────────────
    if orig_str in TRADUCTOR:
        return TRADUCTOR[orig_str]

    for k, v in TRADUCTOR.items():
        if k.lower() == orig_lower:
            return v

    # Si es numérico, resolver el nombre interno en model_names
    nombre_resuelto = None
    if model_names:
        if orig_str.lstrip("-").isdigit():
            idx = int(orig_str)
            if idx in model_names:
                nombre_resuelto = str(model_names[idx])
        elif isinstance(nombre_clase, int) and nombre_clase in model_names:
            nombre_resuelto = str(model_names[nombre_clase])

    # Si se resolvió un nombre interno, verificar nuevamente en Overrides y Traductor
    if nombre_resuelto:
        resuelto_lower = nombre_resuelto.lower()

        # Chequeo en Overrides para el nombre resuelto
        if nombre_resuelto in CORRECCIONES_OVERRIDE:
            return CORRECCIONES_OVERRIDE[nombre_resuelto]
        for k, v in CORRECCIONES_OVERRIDE.items():
            if k.lower() == resuelto_lower:
                return v

        # Chequeo en Traductor para el nombre resuelto
        if nombre_resuelto in TRADUCTOR:
            return TRADUCTOR[nombre_resuelto]
        for k, v in TRADUCTOR.items():
            if k.lower() == resuelto_lower:
                return v

        # Si el modelo tiene nombre pero aún no lo mapeamos:
        return f"⚠️ Nueva Señal: {nombre_resuelto}"

    # ──────────────────────────────────────────────────────────────────────────
    # PASO 3: Fallback de Alerta — Clase no mapeada
    # ──────────────────────────────────────────────────────────────────────────
    return f"⚠️ Nueva Señal: {orig_str}"


def get_color(nombre_es: str) -> tuple:
    """
    Retorna el color BGR para OpenCV.
    Las '⚠️ Nueva Señal' devuelven FUCSIA BRILLANTE (255, 0, 255) para resaltar
    inmediatamente durante las pruebas.
    """
    if nombre_es in COLOR_MAP:
        return COLOR_MAP[nombre_es]
    if "Nueva Señal" in nombre_es or nombre_es.startswith("⚠️"):
        return COLOR_MAP["⚠️ Nueva Señal"]
    return COLOR_MAP["⚠️ Nueva Señal"]


def dibujar_detecciones(imagen_bgr: np.ndarray, results) -> tuple[np.ndarray, list[dict]]:
    """
    Dibuja bounding boxes modernos con esquinas decorativas y etiquetas en español.
    Retorna la imagen anotada y la lista estructurada de detecciones.
    """
    detecciones = []
    img = imagen_bgr.copy()

    for result in results:
        boxes = result.boxes
        nombres = result.names  # Dict del modelo: {id: nombre}

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Obtener el nombre original desde la lista interna del modelo
            nombre_original = nombres.get(cls_id, str(cls_id)) if nombres else str(cls_id)

            # Traducir aplicando la jerarquía (Overrides -> Traductor -> Alerta)
            nombre_es = traducir_clase(nombre_original, model_names=nombres)
            color_bgr = get_color(nombre_es)

            # ── Bounding Box ──
            thickness = 3
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)

            # ── Esquinas decorativas de alta definición ──
            corner_len = min(22, (x2 - x1) // 4, (y2 - y1) // 4)
            cth = thickness + 2
            # Superior-izquierda
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), color_bgr, cth)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), color_bgr, cth)
            # Superior-derecha
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), color_bgr, cth)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), color_bgr, cth)
            # Inferior-izquierda
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), color_bgr, cth)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), color_bgr, cth)
            # Inferior-derecha
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color_bgr, cth)
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color_bgr, cth)

            # ── Etiqueta con fondo ──
            label_cv = f"{nombre_es}  {conf:.0%}"
            font_scale = 0.58
            font_thick = 1
            (tw, th), baseline = cv2.getTextSize(
                label_cv, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thick
            )
            pad = 6
            lx1 = x1
            ly1 = max(y1 - th - pad * 2, 0)
            lx2 = x1 + tw + pad * 2
            ly2 = max(y1, th + pad * 2)

            overlay = img.copy()
            cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color_bgr, -1)
            cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

            cv2.putText(
                img, label_cv,
                (lx1 + pad, ly2 - pad),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA,
            )

            detecciones.append({
                "clase_original": nombre_original,
                "clase_es":       nombre_es,
                "confianza":      conf,
                "bbox":           (x1, y1, x2, y2),
            })

    return img, detecciones


def mostrar_resumen(detecciones: list[dict]) -> None:
    """Muestra tarjetas y estadísticas estructuradas del resultado del análisis."""
    if not detecciones:
        st.markdown("""
        <div class="alert-info">
            🔍 No se detectaron señales de tránsito en este archivo.
            Prueba ajustando el umbral de confianza en la barra lateral o con otra toma.
        </div>
        """, unsafe_allow_html=True)
        return

    conteo: Counter = Counter(d["clase_es"] for d in detecciones)
    conf_por_clase: dict[str, list] = {}
    orig_por_clase: dict[str, str]  = {}
    for d in detecciones:
        conf_por_clase.setdefault(d["clase_es"], []).append(d["confianza"])
        orig_por_clase.setdefault(d["clase_es"], d["clase_original"])

    # ── Alerta si hay señales no mapeadas (para diagnóstico inmediato) ────────
    nuevas = {k: v for k, v in conteo.items() if "Nueva Señal" in k}
    if nuevas:
        ids_pendientes = ", ".join(f'"{orig_por_clase[k]}"' for k in sorted(nuevas.keys()))
        st.markdown(f"""
        <div style="background:rgba(255, 0, 128, 0.08); border:1.5px solid rgba(255, 0, 128, 0.35);
                    border-radius:16px; padding:1.2rem; margin-bottom:1.2rem;">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
                <span style="font-size:1.4rem;">⚠️</span>
                <strong style="color:#d9006c; font-size:1.02rem;">Nuevas Señales Detectadas (Sin Mapear)</strong>
            </div>
            <p style="font-size:0.88rem; color:#4a4a4e; margin:0 0 0.5rem 0;">
                El modelo detectó clases que aún no están asignadas en tu archivo <code>config_senales.py</code>.
                Aparecen resaltadas en <strong>fucsia brillante</strong> en la vista previa.
            </p>
            <div style="display:flex; flex-wrap:wrap; gap:0.4rem; align-items:center;">
                <span style="font-size:0.83rem; font-weight:600; color:#1f2937;">Claves o IDs para agregar:</span>
                <code style="background:white; border:1px solid rgba(255,0,128,0.25); padding:3px 8px; border-radius:8px; font-weight:700; color:#d9006c;">
                    {ids_pendientes}
                </code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Métricas principales ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔎 Total de Detecciones", len(detecciones))
    with col2:
        st.metric("📋 Señales Únicas", len(conteo))
    with col3:
        avg_conf = np.mean([d["confianza"] for d in detecciones])
        st.metric("🎯 Confianza Promedio", f"{avg_conf:.1%}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📌 Detalle de Señales Detectadas")

    # ── Tarjetas individuales con estilo iOS ──
    for clase_es, cantidad in sorted(conteo.items(), key=lambda x: -x[1]):
        avg_c = np.mean(conf_por_clase[clase_es])

        if "Nueva Señal" in clase_es:
            emoji = "⚠️"
            border_style = "border-color: rgba(255, 0, 255, 0.45); background: rgba(255, 0, 255, 0.04);"
        else:
            emoji = EMOJI_MAP.get(clase_es, "🔷")
            border_style = ""

        if avg_c >= 0.75:
            conf_class = "conf-high"
        elif avg_c >= 0.50:
            conf_class = "conf-medium"
        else:
            conf_class = "conf-low"

        st.markdown(f"""
        <div class="signal-card" style="{border_style}">
            <div style="display:flex; align-items:center; gap:0.85rem;">
                <span style="font-size:1.6rem;">{emoji}</span>
                <div>
                    <div class="signal-name">{clase_es}</div>
                    <div class="signal-conf">Confianza promedio: <span class="{conf_class}">{avg_c:.1%}</span></div>
                </div>
            </div>
            <div class="signal-badge">×{cantidad}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENCABEZADO PRINCIPAL (HERO)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-title">
    <div style="text-align:center; margin-bottom:0.4rem;">
        <span style="font-size:3.2rem;">🚦</span>
    </div>
    <h1>DetectaVial Ecuador</h1>
</div>
<p class="hero-subtitle">
    Sistema inteligente de detección y clasificación de señales de tránsito mediante Visión Artificial
</p>
<div style="text-align:center; margin-bottom:1.5rem;">
    <span class="hero-badge">🤖 Ultralytics YOLOv8 · Normativa MTOP / INEN</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BARRA LATERAL (SIDEBAR)
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div style="font-size:1.2rem; font-weight:700; color:#111827; margin-bottom:0.3rem;">🎛️ Panel de Control</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.83rem; color:#6b7280; margin-bottom:1.3rem;">Selecciona el modo y ajusta los parámetros</div>', unsafe_allow_html=True)

    # Selector de modo
    modo = st.radio(
        "Tipo de entrada",
        options=["📷  Subir Imagen", "🎬  Subir Video"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Umbral de confianza
    st.markdown("**🎯 Umbral de Confianza**")
    confianza = st.slider(
        "Confianza mínima",
        min_value=0.10,
        max_value=0.95,
        value=0.35,
        step=0.05,
        label_visibility="collapsed",
    )
    st.caption(f"Filtro activo: solo detecciones ≥ {confianza:.0%}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Estado del modelo
    st.markdown("**⚙️ Estado del Sistema**")
    if Path(MODEL_PATH).exists() and YOLO_AVAILABLE:
        st.markdown("""
        <div class="alert-success">
            ✅ Modelo YOLOv8 listo (best.pt)
        </div>
        """, unsafe_allow_html=True)
    elif not YOLO_AVAILABLE:
        st.error("⚠️ Ultralytics no instalado:\n`pip install ultralytics`")
    else:
        st.error(f"❌ No se encontró '{MODEL_PATH}' en la carpeta actual")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Catálogo de señales soportadas
    with st.expander("📚 Catálogo de Señales Mapeadas"):
        st.caption(f"Traductor: {len(TRADUCTOR)} entradas")
        st.caption(f"Overrides activos: {len(CORRECCIONES_OVERRIDE)} reglas")
        for clave, valor in sorted(TRADUCTOR.items(), key=lambda x: x[1]):
            emoji = EMOJI_MAP.get(valor, "🔷")
            st.caption(f"{emoji} **{valor}**  *(clave: `{clave}`)*")


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DEL MODELO YOLOv8
# ══════════════════════════════════════════════════════════════════════════════

modelo = cargar_modelo(MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
#  MODO 1: SUBIR IMAGEN
# ══════════════════════════════════════════════════════════════════════════════

if "📷" in modo:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📷 Detección en Imágenes")
    st.caption("Carga una imagen en formato JPG, PNG o WebP para detectar señales de tránsito.")

    archivo = st.file_uploader(
        "Arrastra o selecciona una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo is not None:
        pil_img = Image.open(archivo).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        col_orig, col_result = st.columns(2, gap="large")

        with col_orig:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🖼️ Imagen Original")
            st.image(pil_img, use_container_width=True, caption="Imagen cargada")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Resultado del Análisis")

            if modelo is None:
                st.error("❌ El archivo best.pt no está disponible o ultralytics no está instalado.")
            else:
                with st.spinner("🧠 Analizando señales de tránsito con YOLOv8..."):
                    t0 = time.time()
                    results = modelo.predict(
                        source=img_bgr,
                        conf=confianza,
                        verbose=False,
                    )
                    elapsed = time.time() - t0

                img_anotada, detecciones = dibujar_detecciones(img_bgr, results)
                img_rgb = cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB)

                st.image(img_rgb, use_container_width=True, caption=f"Tiempo de inferencia: {elapsed:.2f}s")

            st.markdown('</div>', unsafe_allow_html=True)

        # Resumen estructurado
        if modelo is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Resumen de Señales Detectadas")
            mostrar_resumen(detecciones)
            st.markdown('</div>', unsafe_allow_html=True)

            # Botón de descarga de la imagen resultante
            if detecciones:
                img_rgb_dl = cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB)
                pil_result = Image.fromarray(img_rgb_dl)
                buf = io.BytesIO()
                pil_result.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Descargar Imagen Anotada (PNG)",
                    data=buf.getvalue(),
                    file_name="deteccion_senales_ecuador.png",
                    mime="image/png",
                )

    else:
        st.markdown("""
        <div style="text-align:center; padding:3.5rem 1rem; color:#9ca3af;">
            <div style="font-size:3.8rem; margin-bottom:0.8rem;">📂</div>
            <div style="font-size:1.15rem; font-weight:600; color:#4b5563; margin-bottom:0.3rem;">
                Ninguna imagen seleccionada
            </div>
            <div style="font-size:0.9rem;">
                Arrastra una foto o haz clic en el recuadro superior para comenzar
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODO 2: SUBIR VIDEO
# ══════════════════════════════════════════════════════════════════════════════

elif "🎬" in modo:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎬 Detección en Video")
    st.caption("Sube un video en formato MP4, MOV o AVI para analizar frame a frame las señales viales.")
    st.info("💡 **Consejo de rendimiento:** Para resultados más rápidos, utiliza clips cortos (de 10 a 60 segundos).")

    archivo_video = st.file_uploader(
        "Arrastra o selecciona un video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(archivo_video.read())
            ruta_entrada = tmp_in.name

        ruta_salida = ruta_entrada.replace(".mp4", "_detectado.mp4")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📹 Video Original")
        st.video(ruta_entrada)
        st.markdown('</div>', unsafe_allow_html=True)

        if modelo is None:
            st.error("❌ El archivo best.pt no está disponible o ultralytics no está instalado.")
        else:
            if st.button("▶️  Iniciar Detección en Video"):
                cap = cv2.VideoCapture(ruta_entrada)
                fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
                ancho   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                alto    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### ⚙️ Procesando Fotogramas...")
                barra     = st.progress(0.0)
                estado    = st.empty()
                col_m1, col_m2 = st.columns(2)
                met_frame = col_m1.empty()
                met_det   = col_m2.empty()

                todas_detecciones: list[dict] = []
                frame_idx = 0
                t0 = time.time()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = modelo.predict(
                        source=frame,
                        conf=confianza,
                        verbose=False,
                    )
                    frame_anotado, dets = dibujar_detecciones(frame, results)
                    todas_detecciones.extend(dets)
                    writer.write(frame_anotado)

                    frame_idx += 1
                    progreso = frame_idx / max(total_f, 1)
                    barra.progress(min(progreso, 1.0))
                    elapsed = time.time() - t0
                    fps_proc = frame_idx / elapsed if elapsed > 0 else 0

                    estado.caption(
                        f"Fotograma {frame_idx}/{total_f} · {fps_proc:.1f} fps · Tiempo: {elapsed:.1f}s"
                    )
                    met_frame.metric("🎞️ Frames Procesados", f"{frame_idx}/{total_f}")
                    met_det.metric("🔍 Detecciones Acumuladas", len(todas_detecciones))

                cap.release()
                writer.release()
                st.markdown('</div>', unsafe_allow_html=True)

                # Mostrar video procesado
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### ✅ Video Anotado Resultante")

                if os.path.exists(ruta_salida):
                    st.video(ruta_salida)

                    with open(ruta_salida, "rb") as f:
                        st.download_button(
                            label="⬇️ Descargar Video Anotado (MP4)",
                            data=f.read(),
                            file_name="video_senales_ecuador.mp4",
                            mime="video/mp4",
                        )
                else:
                    st.error("No se pudo generar el video de salida.")

                st.markdown('</div>', unsafe_allow_html=True)

                # Resumen general del video
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📊 Resumen Global del Video")
                mostrar_resumen(todas_detecciones)
                st.markdown('</div>', unsafe_allow_html=True)

                # Limpieza de temporales
                try:
                    os.unlink(ruta_entrada)
                except Exception:
                    pass

    else:
        st.markdown("""
        <div style="text-align:center; padding:3.5rem 1rem; color:#9ca3af;">
            <div style="font-size:3.8rem; margin-bottom:0.8rem;">🎞️</div>
            <div style="font-size:1.15rem; font-weight:600; color:#4b5563; margin-bottom:0.3rem;">
                Ningún video seleccionado
            </div>
            <div style="font-size:0.9rem;">
                Carga un video para procesar y clasificar señales de tránsito frame a frame
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PIE DE PÁGINA (FOOTER)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1.5rem; color:#9ca3af; font-size:0.84rem;">
    🚦 <strong>DetectaVial Ecuador</strong> · Normativa MTOP / INEN<br>
    Desarrollado con <strong>Streamlit</strong> + <strong>Ultralytics YOLOv8</strong>
</div>
""", unsafe_allow_html=True)
