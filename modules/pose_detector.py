"""
Módulo: Detección de Postura
Entidad: DetectorPostura
Compatible con mediapipe 0.10.35 (API Tasks)
"""

import numpy as np
from typing import Optional
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python import BaseOptions
import urllib.request
import os


# Conexiones para dibujar el esqueleto manualmente
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

RUTA_MODELO_MP = "data/model/pose_landmarker.task"
URL_MODELO_MP = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


class DetectorPostura:

    LANDMARKS_RELEVANTES = {
        "nariz": 0,
        "hombro_izq": 11, "hombro_der": 12,
        "codo_izq": 13,   "codo_der": 14,
        "muneca_izq": 15, "muneca_der": 16,
        "cadera_izq": 23, "cadera_der": 24,
        "rodilla_izq": 25, "rodilla_der": 26,
        "tobillo_izq": 27, "tobillo_der": 28,
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self._descargar_modelo()
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=RUTA_MODELO_MP),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._ultimo_resultado = None

    def _descargar_modelo(self) -> None:
        if os.path.exists(RUTA_MODELO_MP):
            return
        os.makedirs(os.path.dirname(RUTA_MODELO_MP), exist_ok=True)
        print(f"[MediaPipe] Descargando modelo pose_landmarker_lite.task ...")
        urllib.request.urlretrieve(URL_MODELO_MP, RUTA_MODELO_MP)
        print(f"[MediaPipe] Modelo descargado en: {RUTA_MODELO_MP}")

    def detectar(self, frame_rgb: np.ndarray) -> Optional[object]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = self._landmarker.detect(mp_image)
        self._ultimo_resultado = resultado
        if resultado.pose_landmarks and len(resultado.pose_landmarks) > 0:
            return resultado
        return None

    def extraer_landmarks(self, resultado) -> Optional[np.ndarray]:
        if resultado is None or not resultado.pose_landmarks:
            return None
        landmarks = []
        for lm in resultado.pose_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(landmarks, dtype=np.float32)

    def extraer_landmarks_relevantes(self, resultado) -> Optional[np.ndarray]:
        if resultado is None or not resultado.pose_landmarks:
            return None
        lms = resultado.pose_landmarks[0]
        vector = []
        for idx in self.LANDMARKS_RELEVANTES.values():
            lm = lms[idx]
            vector.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(vector, dtype=np.float32)

    def usuario_visible(self, resultado) -> bool:
        if resultado is None or not resultado.pose_landmarks:
            return False
        lms = resultado.pose_landmarks[0]
        for idx in self.LANDMARKS_RELEVANTES.values():
            if lms[idx].visibility < 0.5:
                return False
        return True

    def dibujar_landmarks(
        self,
        frame_bgr: np.ndarray,
        resultado,
        color_puntos: tuple = (0, 255, 0),
        color_conexiones: tuple = (255, 255, 255),
    ) -> np.ndarray:
        if resultado is None or not resultado.pose_landmarks:
            return frame_bgr

        lms = resultado.pose_landmarks[0]
        h, w = frame_bgr.shape[:2]

        # Dibujar conexiones
        for a, b in POSE_CONNECTIONS:
            if a < len(lms) and b < len(lms):
                x1, y1 = int(lms[a].x * w), int(lms[a].y * h)
                x2, y2 = int(lms[b].x * w), int(lms[b].y * h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), color_conexiones, 2)

        # Dibujar puntos
        for lm in lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 4, color_puntos, -1)

        return frame_bgr

    def dibujar_overlay_justdance(
        self, frame_bgr: np.ndarray, resultado, visible: bool
    ) -> np.ndarray:
        frame = self.dibujar_landmarks(frame_bgr, resultado)
        color_estado = (0, 255, 0) if visible else (0, 0, 255)
        texto_estado = "DETECTADO" if visible else "AJUSTA POSICION"
        cv2.putText(
            frame, texto_estado, (10, 40),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, color_estado, 2, cv2.LINE_AA,
        )
        return frame

    def cerrar(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "DetectorPostura":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cerrar()