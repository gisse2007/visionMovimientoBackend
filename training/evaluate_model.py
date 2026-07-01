"""
Training: Evaluación del Modelo
Corregido: usa el mismo X/y que el entrenamiento para evaluación consistente.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUTA_X      = "data/model/X_preprocesado.npy"
RUTA_Y      = "data/model/y_preprocesado.npy"
RUTA_MODELO = "data/model/modelo_baile.keras"
RUTA_CLASES = "data/model/clases.json"
RUTA_SALIDA = "data/model"


def evaluar():
    print("\n=== Evaluación del Modelo ===\n")

    # Cargar datos y modelo
    X = np.load(RUTA_X)
    y_str = np.load(RUTA_Y)
    modelo = tf.keras.models.load_model(RUTA_MODELO)

    with open(RUTA_CLASES, "r") as f:
        clases = json.load(f)

    print(f"  Modelo: {RUTA_MODELO}")
    print(f"  Clases: {clases}")
    print(f"  Total muestras: {len(X)}")

    # Codificar etiquetas con el mismo orden que el entrenamiento
    le = LabelEncoder()
    le.fit(clases)
    y = le.transform(y_str)

    # Mismo split que en train_model.py (random_state=42)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Muestras de prueba: {len(X_test)}\n")

    # Predicciones directas
    y_prob = modelo.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"  ✓ Accuracy general: {acc:.4f} ({acc*100:.2f}%)\n")

    # Reporte
    nombres = le.classes_.tolist()
    reporte_str = classification_report(
        y_test, y_pred, target_names=nombres, zero_division=0
    )
    reporte_dict = classification_report(
        y_test, y_pred, target_names=nombres, output_dict=True, zero_division=0
    )
    print("  Reporte de clasificación:")
    print(reporte_str)

    # Guardar reporte
    ruta_reporte = os.path.join(RUTA_SALIDA, "reporte_evaluacion.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write(f"Accuracy general: {acc:.4f}\n\n{reporte_str}")
    print(f"  Reporte guardado: {ruta_reporte}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(nombres)), yticks=np.arange(len(nombres)),
           xticklabels=nombres, yticklabels=nombres,
           ylabel="Real", xlabel="Predicho", title="Matriz de Confusión")