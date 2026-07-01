"""
Módulo: Modelo de IA
Entidad: ModeloIA
Responsabilidad: Cargar y ejecutar el modelo TensorFlow entrenado.
Spec ref: Sección 6 (Entidades), Sección 14 (entrenar con TensorFlow)
"""

import os
import json
import numpy as np
from typing import Optional, List, Dict, Tuple
import tensorflow as tf


class ModeloIA:
    """
    Carga y ejecuta el modelo entrenado para clasificación de movimientos.
    Solo reconoce movimientos previamente entrenados (regla de negocio, sección 9).
    """

    RUTA_MODELO_DEFAULT = "data/model/modelo_baile.keras"
    RUTA_CLASES_DEFAULT = "data/model/clases.json"

    def __init__(
        self,
        ruta_modelo: str = RUTA_MODELO_DEFAULT,
        ruta_clases: str = RUTA_CLASES_DEFAULT,
        umbral_confianza: float = 0.7,
    ):
        self._ruta_modelo = ruta_modelo
        self._ruta_clases = ruta_clases
        self._umbral_confianza = umbral_confianza
        self._modelo: Optional[tf.keras.Model] = None
        self._clases: List[str] = []
        self._cargado: bool = False

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    def cargar(self) -> bool:
        """
        Carga el modelo entrenado y las etiquetas de clases desde disco.

        Returns:
            True si se cargó correctamente, False si no existe aún.
        """
        if not os.path.exists(self._ruta_modelo):
            print(f"[ModeloIA] Modelo no encontrado en: {self._ruta_modelo}")
            print("[ModeloIA] Primero ejecuta el entrenamiento.")
            return False

        if not os.path.exists(self._ruta_clases):
            print(f"[ModeloIA] Archivo de clases no encontrado: {self._ruta_clases}")
            return False

        self._modelo = tf.keras.models.load_model(self._ruta_modelo)

        with open(self._ruta_clases, "r", encoding="utf-8") as f:
            self._clases = json.load(f)

        self._cargado = True
        print(f"[ModeloIA] Modelo cargado. Clases: {self._clases}")
        return True

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predecir(self, vector_landmarks: np.ndarray) -> Dict:
        """
        Clasifica un vector de landmarks y devuelve el movimiento detectado.

        Args:
            vector_landmarks: np.ndarray de forma (N,) con los landmarks aplanados.

        Returns:
            Dict con:
              - movimiento: str (nombre del movimiento o "Desconocido")
              - confianza: float (0.0 - 1.0)
              - todas_probabilidades: Dict[str, float]
              - reconocido: bool
        """
        if not self._cargado or self._modelo is None:
            return self._resultado_vacio("Modelo no cargado")

        entrada = vector_landmarks.reshape(1, -1).astype(np.float32)
        probabilidades = self._modelo.predict(entrada, verbose=0)[0]

        idx_max = int(np.argmax(probabilidades))
        confianza = float(probabilidades[idx_max])

        reconocido = confianza >= self._umbral_confianza
        movimiento = self._clases[idx_max] if reconocido else "Desconocido"

        todas = {
            self._clases[i]: float(probabilidades[i])
            for i in range(len(self._clases))
        }

        return {
            "movimiento": movimiento,
            "confianza": confianza,
            "todas_probabilidades": todas,
            "reconocido": reconocido,
        }

    def predecir_secuencia(self, secuencia: np.ndarray) -> Dict:
        """
        Predice sobre una secuencia de frames (para modelos LSTM).

        Args:
            secuencia: np.ndarray de forma (frames, features)

        Returns:
            Mismo formato que predecir().
        """
        if not self._cargado or self._modelo is None:
            return self._resultado_vacio("Modelo no cargado")

        entrada = secuencia.reshape(1, *secuencia.shape).astype(np.float32)
        probabilidades = self._modelo.predict(entrada, verbose=0)[0]

        idx_max = int(np.argmax(probabilidades))
        confianza = float(probabilidades[idx_max])
        reconocido = confianza >= self._umbral_confianza
        movimiento = self._clases[idx_max] if reconocido else "Desconocido"

        todas = {
            self._clases[i]: float(probabilidades[i])
            for i in range(len(self._clases))
        }

        return {
            "movimiento": movimiento,
            "confianza": confianza,
            "todas_probabilidades": todas,
            "reconocido": reconocido,
        }

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    @property
    def esta_cargado(self) -> bool:
        return self._cargado

    @property
    def clases(self) -> List[str]:
        return self._clases.copy()

    @property
    def umbral_confianza(self) -> float:
        return self._umbral_confianza

    @umbral_confianza.setter
    def umbral_confianza(self, valor: float) -> None:
        if not 0.0 <= valor <= 1.0:
            raise ValueError("El umbral debe estar entre 0.0 y 1.0")
        self._umbral_confianza = valor

    # ------------------------------------------------------------------
    # Privados
    # ------------------------------------------------------------------

    def _resultado_vacio(self, motivo: str) -> Dict:
        return {
            "movimiento": "Desconocido",
            "confianza": 0.0,
            "todas_probabilidades": {},
            "reconocido": False,
            "motivo": motivo,
        }
