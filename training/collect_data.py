"""
Training: Recolección de Dataset
Responsabilidad: Capturar landmarks de poses para construir el dataset de entrenamiento.
Spec ref: Sección 15 paso 3 (Recolección de datos), Pipeline ML sección 14
Uso:
    python training/collect_data.py --movimiento "salto" --muestras 200
"""

import cv2
import numpy as np
import os
import sys
import argparse
import json
from datetime import datetime

# Asegura que el módulo backend sea importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.camera import Camara
from modules.pose_detector import DetectorPostura


RUTA_DATOS = "data/processed"
RUTA_CLASES = "data/model/clases.json"


def recolectar(nombre_movimiento: str, n_muestras: int = 200) -> None:
    """
    Captura landmarks de un movimiento específico y los guarda como .npy.

    Args:
        nombre_movimiento: Nombre del movimiento (ej: "salto", "giro").
        n_muestras: Cantidad de frames de landmarks a capturar.
    """
    os.makedirs(RUTA_DATOS, exist_ok=True)
    os.makedirs("data/model", exist_ok=True)

    ruta_movimiento = os.path.join(RUTA_DATOS, nombre_movimiento)
    os.makedirs(ruta_movimiento, exist_ok=True)

    datos = []

    print(f"\n=== Recolectando datos para: '{nombre_movimiento}' ===")
    print(f"Se capturarán {n_muestras} muestras.")
    print("Presiona ESPACIO para iniciar. Presiona 'q' para salir.\n")

    with Camara(device_index=0) as cam:
        with DetectorPostura() as detector:
            cam.iniciar()
            capturando = False
            contador = 0

            while True:
                ok, frame_bgr = cam.leer_frame_espejo()
                if not ok:
                    break

                frame_rgb = Camara.a_rgb(frame_bgr)
                resultado = detector.detectar(frame_rgb)
                frame_display = detector.dibujar_landmarks(frame_bgr.copy(), resultado)

                estado = f"Capturando: {contador}/{n_muestras}" if capturando else "Listo. Presiona ESPACIO para iniciar"
                color = (0, 255, 0) if capturando else (0, 165, 255)

                cv2.putText(frame_display, f"Movimiento: {nombre_movimiento}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_display, estado, (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if capturando and resultado is not None:
                    landmarks = detector.extraer_landmarks(resultado)
                    if landmarks is not None:
                        datos.append(landmarks)
                        contador += 1

                    if contador >= n_muestras:
                        print(f"[OK] {n_muestras} muestras capturadas.")
                        break

                cv2.imshow(f"Recoleccion - {nombre_movimiento}", frame_display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(" "):
                    capturando = True
                    print("Capturando...")
                elif key == ord("q"):
                    print("Cancelado por el usuario.")
                    break

    cv2.destroyAllWindows()

    if datos:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo = os.path.join(ruta_movimiento, f"{timestamp}.npy")
        np.save(archivo, np.array(datos))
        print(f"[Guardado] {len(datos)} muestras en: {archivo}")
        _actualizar_clases(nombre_movimiento)
    else:
        print("[Aviso] No se guardaron datos.")


def _actualizar_clases(nuevo_movimiento: str) -> None:
    """Agrega el movimiento al archivo de clases si no existe."""
    clases = []
    if os.path.exists(RUTA_CLASES):
        with open(RUTA_CLASES, "r") as f:
            clases = json.load(f)

    if nuevo_movimiento not in clases:
        clases.append(nuevo_movimiento)
        clases.sort()
        with open(RUTA_CLASES, "w") as f:
            json.dump(clases, f, indent=2)
        print(f"[Clases] '{nuevo_movimiento}' agregado. Total: {clases}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolector de datos de baile")
    parser.add_argument("--movimiento", type=str, required=True,
                        help="Nombre del movimiento a capturar (ej: salto, giro)")
    parser.add_argument("--muestras", type=int, default=200,
                        help="Número de frames a capturar (default: 200)")
    args = parser.parse_args()

    recolectar(args.movimiento, args.muestras)
