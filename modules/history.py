"""
Módulo: Historial de Movimientos
Entidad: HistorialMovimientos
Responsabilidad: Almacenar resultados y estadísticas localmente en JSON.
Spec ref: Sección 6, 8, 12 (persistencia local sin base de datos)
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class HistorialMovimientos:
    """
    Almacena resultados y estadísticas en un archivo JSON local.
    Sin base de datos (regla de negocio, sección 14).
    """

    RUTA_DEFAULT = "history/movements.json"

    def __init__(self, ruta_archivo: str = RUTA_DEFAULT):
        self._ruta = ruta_archivo
        self._registros: List[Dict] = []
        self._cargar()

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------

    def registrar(
        self,
        movimiento: str,
        confianza: float,
        reconocido: bool,
        sesion_id: Optional[str] = None,
    ) -> Dict:
        """
        Registra un movimiento detectado con timestamp.

        Args:
            movimiento: Nombre del movimiento.
            confianza: Nivel de confianza (0.0 - 1.0).
            reconocido: Si fue reconocido por el modelo.
            sesion_id: Identificador de sesión opcional.

        Returns:
            El registro creado.
        """
        registro = {
            "id": len(self._registros) + 1,
            "timestamp": datetime.now().isoformat(),
            "movimiento": movimiento,
            "confianza": round(confianza, 4),
            "reconocido": reconocido,
            "sesion_id": sesion_id or self._sesion_actual(),
        }
        self._registros.append(registro)
        self._guardar()
        return registro

    def limpiar_historial(self) -> None:
        """Elimina todos los registros del historial."""
        self._registros = []
        self._guardar()

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def obtener_todos(self) -> List[Dict]:
        return self._registros.copy()

    def obtener_por_sesion(self, sesion_id: str) -> List[Dict]:
        return [r for r in self._registros if r.get("sesion_id") == sesion_id]

    def obtener_ultimos(self, n: int = 20) -> List[Dict]:
        return self._registros[-n:]

    def estadisticas(self) -> Dict:
        """
        Calcula estadísticas del historial.

        Returns:
            Dict con total, por movimiento, confianza promedio, etc.
        """
        if not self._registros:
            return {"total": 0, "movimientos": {}, "confianza_promedio": 0.0}

        total = len(self._registros)
        reconocidos = [r for r in self._registros if r["reconocido"]]
        confianza_promedio = (
            sum(r["confianza"] for r in reconocidos) / len(reconocidos)
            if reconocidos
            else 0.0
        )

        conteo_movimientos: Dict[str, int] = {}
        for r in self._registros:
            mv = r["movimiento"]
            conteo_movimientos[mv] = conteo_movimientos.get(mv, 0) + 1

        return {
            "total": total,
            "reconocidos": len(reconocidos),
            "tasa_reconocimiento": round(len(reconocidos) / total, 4) if total else 0,
            "confianza_promedio": round(confianza_promedio, 4),
            "movimientos": conteo_movimientos,
            "movimiento_mas_frecuente": (
                max(conteo_movimientos, key=conteo_movimientos.get)
                if conteo_movimientos
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Persistencia local
    # ------------------------------------------------------------------

    def _guardar(self) -> None:
        os.makedirs(os.path.dirname(self._ruta), exist_ok=True)
        with open(self._ruta, "w", encoding="utf-8") as f:
            json.dump(self._registros, f, ensure_ascii=False, indent=2)

    def _cargar(self) -> None:
        if os.path.exists(self._ruta):
            with open(self._ruta, "r", encoding="utf-8") as f:
                self._registros = json.load(f)
        else:
            self._registros = []

    @staticmethod
    def _sesion_actual() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def total_registros(self) -> int:
        return len(self._registros)
