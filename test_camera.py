"""
Test standalone: Módulo Camara
Ejecutar desde backend/:
    python test_camera.py
Presionar 'q' para salir.
"""

import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.camera import Camara


def main():
    print("=== Test Módulo Camara ===")
    with Camara(device_index=0, width=640, height=480, fps=30) as cam:
        print(f"Activa: {cam.esta_activa}")
        print(f"Resolución: {cam.resolucion}")
        print(f"FPS: {cam.fps_real}")
        print("Presiona 'q' para salir.")

        while True:
            ok, frame = cam.leer_frame_espejo()
            if not ok or frame is None:
                print("Error al leer frame.")
                break

            cv2.putText(frame, f"MovimientoAI | {cam.resolucion[0]}x{cam.resolucion[1]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Test Camara", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Cámara liberada correctamente.")


if __name__ == "__main__":
    main()
