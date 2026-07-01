"""
Punto de entrada: FastAPI + Uvicorn
Compatible con local (cámara OpenCV) y nube (frames desde navegador)
"""

import cv2
import asyncio
import threading
import base64
import numpy as np
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from modules.camera import Camara
from modules.pose_detector import DetectorPostura
from modules.classifier import ClasificadorMovimiento
from modules.ai_model import ModeloIA
from modules.history import HistorialMovimientos


# ------------------------------------------------------------------
# Estado global
# ------------------------------------------------------------------

camara = Camara(device_index=0, width=640, height=480, fps=30)
detector = DetectorPostura()
modelo_ia = ModeloIA()
clasificador = ClasificadorMovimiento(detector, modelo_ia)
historial = HistorialMovimientos()

reconocimiento_activo = False
ultimo_resultado: dict = {}
lock_resultado = threading.Lock()
MODO_NUBE = False  # Se detecta automáticamente


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODO_NUBE
    modelo_ia.cargar()
    # Detectar si hay cámara disponible
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            MODO_NUBE = False
            cap.release()
        else:
            MODO_NUBE = True
    except Exception:
        MODO_NUBE = True
    print(f"[Sistema] Modo: {'NUBE (sin cámara)' if MODO_NUBE else 'LOCAL (con cámara)'}")
    yield
    camara.detener()
    detector.cerrar()


app = FastAPI(
    title="MovimientoAI API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Modelos de request
# ------------------------------------------------------------------

class FrameRequest(BaseModel):
    frame: str  # base64


# ------------------------------------------------------------------
# Generador de frames para streaming MJPEG (solo local)
# ------------------------------------------------------------------

def generar_frames():
    global ultimo_resultado
    while True:
        if not camara.esta_activa:
            break
        ok, frame_bgr = camara.leer_frame_espejo()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = Camara.a_rgb(frame_bgr)
        resultado_pose = detector.detectar(frame_rgb)
        visible = detector.usuario_visible(resultado_pose)
        frame_display = detector.dibujar_overlay_justdance(
            frame_bgr.copy(), resultado_pose, visible
        )
        if reconocimiento_activo and modelo_ia.esta_cargado and resultado_pose is not None:
            resultado = clasificador.clasificar(frame_rgb)
            with lock_resultado:
                ultimo_resultado = resultado
            movimiento = resultado.get("movimiento_suavizado", "...")
            confianza = resultado.get("confianza", 0.0)
            color_mv = (0, 255, 255) if resultado.get("reconocido") else (128, 128, 128)
            cv2.putText(frame_display, f"Movimiento: {movimiento}", (10, 420),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color_mv, 2)
            cv2.putText(frame_display, f"Confianza: {confianza:.0%}", (10, 455),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        jpeg_bytes = Camara.codificar_jpeg(frame_display, calidad=80)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
        )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/")
def raiz():
    return {"sistema": "MovimientoAI", "estado": "activo", "modo": "nube" if MODO_NUBE else "local"}


@app.get("/health")
def health():
    return {
        "camara_activa": camara.esta_activa,
        "modelo_cargado": modelo_ia.esta_cargado,
        "reconocimiento_activo": reconocimiento_activo,
        "clases_disponibles": modelo_ia.clases,
        "modo": "nube" if MODO_NUBE else "local",
    }


@app.post("/camara/iniciar")
def iniciar_camara():
    if MODO_NUBE:
        return {"ok": True, "mensaje": "Modo nube: usa el endpoint /frame/clasificar"}
    try:
        camara.iniciar()
        return {"ok": True, "mensaje": "Cámara iniciada", "resolucion": camara.resolucion}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/camara/detener")
def detener_camara():
    camara.detener()
    return {"ok": True, "mensaje": "Cámara detenida"}


@app.get("/video/stream")
def stream_video():
    if MODO_NUBE:
        raise HTTPException(status_code=400, detail="Modo nube: no hay stream disponible")
    if not camara.esta_activa:
        raise HTTPException(status_code=400, detail="La cámara no está activa")
    return StreamingResponse(
        generar_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/frame/clasificar")
async def clasificar_frame(request: FrameRequest):
    """
    Recibe un frame en base64 desde el navegador,
    lo procesa con MediaPipe + TensorFlow y devuelve el resultado.
    Usado en modo nube.
    """
    if not modelo_ia.esta_cargado:
        raise HTTPException(status_code=400, detail="Modelo no cargado")

    try:
        # Decodificar base64
        img_data = base64.b64decode(request.frame.split(",")[-1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise HTTPException(status_code=400, detail="Frame inválido")

        frame_rgb = Camara.a_rgb(frame_bgr)
        resultado = clasificador.clasificar(frame_rgb)

        with lock_resultado:
            global ultimo_resultado
            ultimo_resultado = resultado

        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reconocimiento/iniciar")
def iniciar_reconocimiento():
    global reconocimiento_activo
    if not modelo_ia.esta_cargado:
        raise HTTPException(status_code=400, detail="Modelo no cargado")
    reconocimiento_activo = True
    clasificador.resetear_ventana()
    return {"ok": True, "mensaje": "Reconocimiento iniciado"}


@app.post("/reconocimiento/detener")
def detener_reconocimiento():
    global reconocimiento_activo
    reconocimiento_activo = False
    return {"ok": True, "mensaje": "Reconocimiento detenido"}


@app.get("/reconocimiento/resultado")
def obtener_resultado():
    with lock_resultado:
        resultado = ultimo_resultado.copy()
    return resultado


@app.post("/reconocimiento/registrar")
def registrar_movimiento():
    with lock_resultado:
        resultado = ultimo_resultado.copy()
    if not resultado or not resultado.get("reconocido"):
        raise HTTPException(status_code=400, detail="No hay movimiento reconocido")
    registro = historial.registrar(
        movimiento=resultado["movimiento_suavizado"],
        confianza=resultado["confianza"],
        reconocido=resultado["reconocido"],
    )
    return {"ok": True, "registro": registro}


@app.get("/historial")
def obtener_historial(ultimos: int = 50):
    return {
        "registros": historial.obtener_ultimos(ultimos),
        "total": historial.total_registros,
    }


@app.get("/historial/estadisticas")
def obtener_estadisticas():
    return historial.estadisticas()


@app.delete("/historial/limpiar")
def limpiar_historial():
    historial.limpiar_historial()
    return {"ok": True, "mensaje": "Historial limpiado"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)