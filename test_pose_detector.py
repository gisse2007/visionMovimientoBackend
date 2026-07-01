"""
Test standalone: Módulo 2 — DetectorPostura con MediaPipe
Ejecutar desde backend/:
    python test_pose_detector.py
Presionar 'q' para salir.

Muestra:
  - Landmarks del cuerpo dibujados en tiempo real (estilo Just Dance)
  - Estado de visibilidad del usuario
  - Vector de landmarks en consola (para verificar extracción)
"""

import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.camera import Camara
from modules.pose_detector import DetectorPostura


def main() -> None:
    print("=== Test Módulo 2: DetectorPostura (MediaPipe) ===")
    print("Presiona 'q' para salir | 'i' para imprimir landmarks en consola\n")

    frame_count = 0
    mostrar_info = False

    with Camara(device_index=0, width=640, height=480, fps=30) as cam:
        with DetectorPostura(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ) as detector:

            while True:
                ok, frame_bgr = cam.leer_frame_espejo()
                if not ok or frame_bgr is None:
                    print("Error al leer frame.")
                    break

                frame_rgb = Camara.a_rgb(frame_bgr)

                # Detectar postura
                resultado = detector.detectar(frame_rgb)
                visible   = detector.usuario_visible(resultado)

                # Dibujar overlay estilo Just Dance
                frame_display = detector.dibujar_overlay_justdance(
                    frame_bgr.copy(), resultado, visible
                )

                # Info en pantalla
                landmarks_detectados = resultado is not None
                color_lm = (0, 255, 0) if landmarks_detectados else (100, 100, 100)
                cv2.putText(
                    frame_display,
                    f"Landmarks: {'SI' if landmarks_detectados else 'NO'}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_lm, 2,
                )

                # FPS aproximado
                frame_count += 1
                cv2.putText(
                    frame_display,
                    f"Frame: {frame_count}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1,
                )

                # Extraer y mostrar vector si se presiona 'i'
                if mostrar_info and resultado is not None:
                    vector_completo   = detector.extraer_landmarks(resultado)
                    vector_relevante  = detector.extraer_landmarks_relevantes(resultado)

                    if vector_completo is not None:
                        print(f"\n--- Frame {frame_count} ---")
                        print(f"  Vector completo  shape: {vector_completo.shape}")
                        print(f"  Vector relevante shape: {vector_relevante.shape}")
                        print(f"  Primeros 12 valores: {vector_completo[:12].round(4)}")
                        print(f"  Visible completo: {visible}")

                    mostrar_info = False

                cv2.imshow("Test Pose Detector — MovimientoAI", frame_display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("i"):
                    mostrar_info = True

    cv2.destroyAllWindows()
    print("\nDetector cerrado correctamente.")


if __name__ == "__main__":
    main()
