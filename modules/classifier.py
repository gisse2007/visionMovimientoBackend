"""
Módulo: Clasificación de Movimientos
Entidad: ClasificadorMovimiento
Responsabilidad: Orquestar DetectorPostura + ModeloIA para clasificar en tiempo real.
Spec ref: Sección 5, 6, Pipeline ML (sección 14)
"""

import numpy as np
from typing import Optional, Dict, Deque
from collections import deque

from modules.pose_detector import DetectorPostura
from modules.ai_model import ModeloIA


class ClasificadorMovimiento:
    """
    Clasifica el movimiento detectado combinando:
    - DetectorPostura (extracción de landmarks)
    - ModeloIA (inferencia TensorFlow)

    Implementa suavizado por ventana deslizante para resultados estables
    en tiempo real (priorizar rendimiento en tiempo real, sección 14).
    """

    def __init__(
        self,
        detector: DetectorPostura,
        modelo: ModeloIA,
        ventana_suavizado: int = 10,
    ):
        self._detector = detector
        self._modelo = modelo
        self._ventana: Deque[str] = deque(maxlen=ventana_suavizado)
        self._ultimo_resultado: Dict = {}

    # ------------------------------------------------------------------
    # Clasificación principal
    # ------------------------------------------------------------------

    def clasificar(self, frame_rgb: np.ndarray) -> Dict:
        """
        Procesa un frame RGB y devuelve la clasificación del movimiento.

        Flujo:
        1. DetectorPostura extrae landmarks del frame.
        2. Verifica visibilidad del usuario.
        3. ModeloIA predice el movimiento.
        4. Aplica suavizado por ventana.

        Args:
            frame_rgb: Frame en formato RGB.

        Returns:
            Dict con:
              - movimiento: str
              - confianza: float
              - usuario_visible: bool
              - landmarks_detectados: bool
              - movimiento_suavizado: str
        """
        resultado_base = {
            "movimiento": "Sin detección",
            "confianza": 0.0,
            "usuario_visible": False,
            "landmarks_detectados": False,
            "movimiento_suavizado": "Sin detección",
            "todas_probabilidades": {},
            "reconocido": False,
        }

        # Paso 1: Detectar postura
        resultado_pose = self._detector.detectar(frame_rgb)
        if resultado_pose is None:
            self._ultimo_resultado = resultado_base
            return resultado_base

        resultado_base["landmarks_detectados"] = True

        # Paso 2: Verificar visibilidad (regla de negocio, sección 9)
        visible = self._detector.usuario_visible(resultado_pose)
        resultado_base["usuario_visible"] = visible

        # Paso 3: Extraer vector de features
        landmarks = self._detector.extraer_landmarks(resultado_pose)
        if landmarks is None:
            self._ultimo_resultado = resultado_base
            return resultado_base

        # Paso 4: Inferencia
        prediccion = self._modelo.predecir(landmarks)
        resultado_base.update(prediccion)

        # Paso 5: Suavizado temporal
        if prediccion["reconocido"]:
            self._ventana.append(prediccion["movimiento"])

        movimiento_suavizado = self._calcular_suavizado()
        resultado_base["movimiento_suavizado"] = movimiento_suavizado

        self._ultimo_resultado = resultado_base
        return resultado_base

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _calcular_suavizado(self) -> str:
        """
        Devuelve el movimiento más frecuente en la ventana deslizante.
        Reduce falsos positivos en tiempo real.
        """
        if not self._ventana:
            return "Sin detección"

        from collections import Counter
        conteo = Counter(self._ventana)
        return conteo.most_common(1)[0][0]

    def resetear_ventana(self) -> None:
        """Limpia el historial de suavizado."""
        self._ventana.clear()

    @property
    def ultimo_resultado(self) -> Dict:
        return self._ultimo_resultado.copy()

    @property
    def detector(self) -> DetectorPostura:
        return self._detector

    @property
    def modelo(self) -> ModeloIA:
        return self._modelo
