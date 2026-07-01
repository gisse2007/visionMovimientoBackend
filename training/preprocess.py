"""
Training: Preprocesamiento de Datos
Responsabilidad: Cargar, limpiar, normalizar y exportar el dataset final.
Spec ref: Sección 15 paso 4 (Preprocesamiento), Pipeline ML sección 14
Uso:
    python training/preprocess.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import joblib
import matplotlib
matplotlib.use("Agg")  # Evita conflicto circular en Windows
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUTA_DATOS     = "data/processed"
RUTA_MODELO    = "data/model"
RUTA_CLASES    = "data/model/clases.json"
RUTA_SCALER    = "data/model/scaler.pkl"
RUTA_X_OUT     = "data/model/X_preprocesado.npy"
RUTA_Y_OUT     = "data/model/y_preprocesado.npy"


def cargar_datos_crudos() -> tuple[np.ndarray, np.ndarray]:
    """
    Carga todos los archivos .npy de data/processed/<clase>/*.npy

    Returns:
        X: np.ndarray de forma (N, features)
        y: np.ndarray de strings con etiquetas
    """
    if not os.path.exists(RUTA_CLASES):
        raise FileNotFoundError(f"No se encontró {RUTA_CLASES}. Ejecuta collect_data.py primero.")

    with open(RUTA_CLASES, "r") as f:
        clases = json.load(f)

    X, y = [], []
    print("\n[Preprocess] Cargando datos crudos...")

    for clase in clases:
        ruta_clase = os.path.join(RUTA_DATOS, clase)
        if not os.path.exists(ruta_clase):
            print(f"  ⚠ Sin datos para clase: {clase}")
            continue

        archivos = [f for f in os.listdir(ruta_clase) if f.endswith(".npy")]
        n_muestras = 0
        for archivo in archivos:
            datos = np.load(os.path.join(ruta_clase, archivo))
            for muestra in datos:
                X.append(muestra)
                y.append(clase)
                n_muestras += 1

        print(f"  ✓ {clase}: {n_muestras} muestras")

    if not X:
        raise ValueError("No hay datos. Ejecuta collect_data.py primero.")

    return np.array(X, dtype=np.float32), np.array(y)


def limpiar_datos(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Elimina muestras con NaN, Inf o landmarks con visibilidad cero total.

    Returns:
        X, y limpios.
    """
    print("\n[Preprocess] Limpiando datos...")
    n_original = len(X)

    # Eliminar NaN e Inf
    mascara_validos = np.all(np.isfinite(X), axis=1)
    X = X[mascara_validos]
    y = y[mascara_validos]

    n_eliminados = n_original - len(X)
    print(f"  Eliminados por NaN/Inf: {n_eliminados}")
    print(f"  Muestras válidas: {len(X)}")

    return X, y


def normalizar_datos(X: np.ndarray, entrenar_scaler: bool = True) -> np.ndarray:
    """
    Aplica StandardScaler para normalizar los landmarks.
    Guarda el scaler para usar en inferencia.

    Args:
        X: Dataset de features.
        entrenar_scaler: Si True, ajusta y guarda el scaler.

    Returns:
        X normalizado.
    """
    print("\n[Preprocess] Normalizando con StandardScaler...")
    os.makedirs(RUTA_MODELO, exist_ok=True)

    if entrenar_scaler:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        joblib.dump(scaler, RUTA_SCALER)
        print(f"  Scaler guardado en: {RUTA_SCALER}")
    else:
        if not os.path.exists(RUTA_SCALER):
            raise FileNotFoundError("Scaler no encontrado. Ejecuta preprocess.py primero.")
        scaler = joblib.load(RUTA_SCALER)
        X_norm = scaler.transform(X)

    return X_norm.astype(np.float32)


def analizar_distribucion(y: np.ndarray) -> None:
    """Genera y guarda gráfica de distribución de clases."""
    clases, conteos = np.unique(y, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(clases, conteos, color="#7c3aed", edgecolor="#a855f7", linewidth=1.2)
    
    for bar, count in zip(bars, conteos):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_title("Distribución de muestras por clase", fontsize=14, fontweight="bold")
    ax.set_xlabel("Movimiento")
    ax.set_ylabel("Número de muestras")
    ax.grid(axis="y", alpha=0.3)
    
    ruta_grafica = os.path.join(RUTA_MODELO, "distribucion_clases.png")
    plt.tight_layout()
    plt.savefig(ruta_grafica)
    plt.close()
    print(f"  Gráfica guardada en: {ruta_grafica}")

    # Advertir si hay desbalance
    min_c, max_c = conteos.min(), conteos.max()
    if max_c / min_c > 2:
        print(f"  ⚠ DESBALANCE detectado: {min_c} vs {max_c} muestras.")
        print("    Considera recolectar más datos de las clases con menos muestras.")


def exportar_csv(X: np.ndarray, y: np.ndarray) -> None:
    """Exporta el dataset preprocesado como CSV para inspección con pandas."""
    n_features = X.shape[1]
    columnas = [f"f{i}" for i in range(n_features)] + ["etiqueta"]
    df = pd.DataFrame(np.column_stack([X, y]), columns=columnas)
    
    ruta_csv = os.path.join(RUTA_MODELO, "dataset_preprocesado.csv")
    df.to_csv(ruta_csv, index=False)
    print(f"  CSV exportado: {ruta_csv} ({len(df)} filas)")


def preprocesar() -> None:
    print("\n=== Preprocesamiento del Dataset ===")

    # 1. Cargar
    X, y = cargar_datos_crudos()
    print(f"\n  Total: {len(X)} muestras | Features: {X.shape[1]}")

    # 2. Limpiar
    X, y = limpiar_datos(X, y)

    # 3. Analizar distribución
    analizar_distribucion(y)

    # 4. Mezclar aleatoriamente
    X, y = shuffle(X, y, random_state=42)

    # 5. Normalizar
    X_norm = normalizar_datos(X, entrenar_scaler=True)

    # 6. Guardar arrays preprocesados
    os.makedirs(RUTA_MODELO, exist_ok=True)
    np.save(RUTA_X_OUT, X_norm)
    np.save(RUTA_Y_OUT, y)
    print(f"\n  X guardado: {RUTA_X_OUT}")
    print(f"  y guardado: {RUTA_Y_OUT}")

    # 7. Exportar CSV
    exportar_csv(X_norm, y)

    print("\n[OK] Preprocesamiento completado.")
    print("     Siguiente paso: python training/train_model.py")


if __name__ == "__main__":
    preprocesar()
