"""
Training: Entrenamiento del Modelo
Responsabilidad: Cargar el dataset, preprocesar y entrenar el modelo con TensorFlow.
Spec ref: Sección 15 pasos 4 y 5, Pipeline ML sección 14
Uso:
    python training/train_model.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Evita conflicto circular en Windows
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUTA_DATOS = "data/processed"
RUTA_MODELO = "data/model/modelo_baile.keras"
RUTA_CLASES = "data/model/clases.json"
RUTA_GRAFICAS = "data/model"

EPOCAS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2


def cargar_dataset() -> tuple:
    """
    Carga todos los archivos .npy de la carpeta data/processed/.

    Returns:
        Tuple (X, y) donde X son los landmarks y y son las etiquetas.
    """
    X, y = [], []

    if not os.path.exists(RUTA_CLASES):
        raise FileNotFoundError(f"No se encontró {RUTA_CLASES}. Ejecuta collect_data.py primero.")

    with open(RUTA_CLASES, "r") as f:
        clases = json.load(f)

    print(f"[Dataset] Clases encontradas: {clases}")

    for clase in clases:
        ruta_clase = os.path.join(RUTA_DATOS, clase)
        if not os.path.exists(ruta_clase):
            print(f"[Aviso] No hay datos para: {clase}")
            continue

        archivos = [f for f in os.listdir(ruta_clase) if f.endswith(".npy")]
        for archivo in archivos:
            datos = np.load(os.path.join(ruta_clase, archivo))
            for muestra in datos:
                X.append(muestra)
                y.append(clase)

        print(f"  → {clase}: {sum(1 for label in y if label == clase)} muestras")

    if not X:
        raise ValueError("No hay datos de entrenamiento. Ejecuta collect_data.py primero.")

    return np.array(X, dtype=np.float32), np.array(y)


def construir_modelo(n_features: int, n_clases: int) -> keras.Model:
    """
    Construye la arquitectura de la red neuronal.
    Dense layers con Dropout para clasificación de poses.
    """
    modelo = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(n_clases, activation="softmax"),
    ], name="clasificador_baile")

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    modelo.summary()
    return modelo


def graficar_entrenamiento(historia) -> None:
    """Genera y guarda gráficas de accuracy y loss del entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(historia.history["accuracy"], label="Train Accuracy")
    ax1.plot(historia.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Accuracy durante el entrenamiento")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(historia.history["loss"], label="Train Loss")
    ax2.plot(historia.history["val_loss"], label="Val Loss")
    ax2.set_title("Loss durante el entrenamiento")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    ruta_grafica = os.path.join(RUTA_GRAFICAS, "entrenamiento.png")
    plt.tight_layout()
    plt.savefig(ruta_grafica)
    plt.close()
    print(f"[Gráfica] Guardada en: {ruta_grafica}")


def entrenar() -> None:
    print("\n=== Entrenamiento del Modelo de Baile ===\n")

    # 1. Cargar datos
    X, y_str = cargar_dataset()
    print(f"\n[Dataset] Total muestras: {len(X)} | Features por muestra: {X.shape[1]}")

    # 2. Codificar etiquetas
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    clases = le.classes_.tolist()

    # Actualizar clases con orden del LabelEncoder
    with open(RUTA_CLASES, "w") as f:
        json.dump(clases, f, indent=2)
    print(f"[Clases] Orden final: {clases}")

    # 3. Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    print(f"[Split] Train: {len(X_train)} | Validación: {len(X_val)}")

    # 4. Construir modelo
    modelo = construir_modelo(n_features=X.shape[1], n_clases=len(clases))

    # 5. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            RUTA_MODELO, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    # 6. Entrenar
    print(f"\n[Entrenando] {EPOCAS} épocas máx | batch={BATCH_SIZE}\n")
    historia = modelo.fit(
        X_train, y_train,
        epochs=EPOCAS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluar
    loss, acc = modelo.evaluate(X_val, y_val, verbose=0)
    print(f"\n[Resultado] Val Accuracy: {acc:.4f} | Val Loss: {loss:.4f}")

    # 8. Guardar gráficas
    graficar_entrenamiento(historia)

    print(f"\n[OK] Modelo guardado en: {RUTA_MODELO}")
    print(f"[OK] Clases guardadas en: {RUTA_CLASES}")


if __name__ == "__main__":
    entrenar()
