"""
Módulo: Captura de Video
Entidad: Camara
Responsabilidad: Gestionar la captura de frames desde la cámara física usando OpenCV.
Spec ref: Sección 5 (Módulos), Sección 6 (Entidades), Sección 9 (Reglas de negocio)
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class Camara:
    """
    Encargada de capturar imágenes y video desde el dispositivo físico.

    Reglas de negocio aplicadas:
    - La cámara debe estar activa (sección 9).
    - El reconocimiento debe ejecutarse en tiempo real (sección 9).
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._capture: Optional[cv2.VideoCapture] = None
        self._is_active: bool = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        if self._is_active:
            return
        self._capture = cv2.VideoCapture(self._device_index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara con índice {self._device_index}."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture.set(cv2.CAP_PROP_FPS, self._fps)
        self._is_active = True

    def detener(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._is_active = False

    # ------------------------------------------------------------------
    # Captura de frames
    # ------------------------------------------------------------------

    def leer_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        self._verificar_activa()
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        return True, frame

    def leer_frame_rgb(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame_bgr = self.leer_frame()
        if not ok or frame_bgr is None:
            return False, None
        return True, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def leer_frame_espejo(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.leer_frame()
        if not ok or frame is None:
            return False, None
        return True, cv2.flip(frame, 1)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def redimensionar(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(frame, (width, height))

    @staticmethod
    def a_rgb(frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def a_bgr(frame_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def codificar_jpeg(frame: np.ndarray, calidad: int = 85) -> bytes:
        params = [cv2.IMWRITE_JPEG_QUALITY, calidad]
        _, buffer = cv2.imencode(".jpg", frame, params)
        return buffer.tobytes()

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    @property
    def esta_activa(self) -> bool:
        return (
            self._is_active
            and self._capture is not None
            and self._capture.isOpened()
        )

    @property
    def resolucion(self) -> Tuple[int, int]:
        self._verificar_activa()
        w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    @property
    def fps_real(self) -> float:
        self._verificar_activa()
        return self._capture.get(cv2.CAP_PROP_FPS)

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camara":
        self.iniciar()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detener()

    # ------------------------------------------------------------------
    # Privados
    # ------------------------------------------------------------------

    def _verificar_activa(self) -> None:
        if not self.esta_activa:
            raise RuntimeError(
                "La cámara no está activa. Llame a iniciar() primero."
            )
