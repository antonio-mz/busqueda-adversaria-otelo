import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.utils import set_random_seed
SEMILLA = 394867
set_random_seed(SEMILLA)

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from keras import Sequential, Input
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from keras.optimizers import SGD

# LECTURA
# Every file holds whole games; game numbers must not repeat across files.
FICHEROS_PARTIDAS = ["partidas_ia_vs_ia.csv", "partidas_extra.csv"]
tablas = [pd.read_csv(f) for f in FICHEROS_PARTIDAS if os.path.exists(f)]
for i, tabla_a in enumerate(tablas):
    for tabla_b in tablas[i + 1:]:
        repetidas = set(tabla_a['partida']) & set(tabla_b['partida'])
        assert not repetidas, f"game numbers repeated across files: {sorted(repetidas)[:5]}"
partidas_otelo = pd.concat(tablas, ignore_index=True)
print(f"{partidas_otelo['partida'].nunique()} partidas, {len(partidas_otelo)} posiciones")
print(partidas_otelo.head())

# SEPARACION ATRIBUTOS Y OBJETIVO
# The slice stops at 'turno', so 'ganador' and 'partida' are never used as features.
atributos = partidas_otelo.loc[:, '(0, 0)':'turno']
print(atributos.head())

ganador = partidas_otelo.loc[:, 'ganador']
objetivo = np.zeros_like(ganador, dtype=float)
objetivo[ganador == 1] = -1.0  # Negras ganan
objetivo[ganador == 2] = 1.0   # Blancas ganan
print(objetivo[:5])

partidas = partidas_otelo.loc[:, 'partida']


def separar_por_partidas(indices, grupos, proporcion):
    """Split row indices so that every game lands entirely on one side."""
    separador = GroupShuffleSplit(n_splits=1, test_size=proporcion, random_state=SEMILLA)
    dentro, fuera = next(separador.split(indices, groups=grupos))
    return indices[dentro], indices[fuera]


# SEPARACION TRAIN / VALIDACION / TEST POR PARTIDAS COMPLETAS
# Positions of one game are near-duplicates sharing the same label, so a random
# split by position would leak games into the test set.
todos = np.arange(len(partidas_otelo))
indices_resto, indices_prueba = separar_por_partidas(todos, partidas, 0.2)
indices_entrenamiento, indices_validacion = separar_por_partidas(
    indices_resto, partidas.iloc[indices_resto], 0.2
)

conjuntos = {
    'entrenamiento': indices_entrenamiento,
    'validacion': indices_validacion,
    'prueba': indices_prueba,
}
partidas_por_conjunto = {nombre: set(partidas.iloc[idx]) for nombre, idx in conjuntos.items()}
assert partidas_por_conjunto['entrenamiento'].isdisjoint(partidas_por_conjunto['validacion'])
assert partidas_por_conjunto['entrenamiento'].isdisjoint(partidas_por_conjunto['prueba'])
assert partidas_por_conjunto['validacion'].isdisjoint(partidas_por_conjunto['prueba'])

for nombre, idx in conjuntos.items():
    print(f"{nombre}: {len(partidas_por_conjunto[nombre])} partidas, {len(idx)} posiciones")

def aumentar_con_simetrias(atributos, objetivo):
    """Add the 8 symmetries of the board (4 rotations x optional mirror).

    Othello rules are invariant under rotating or mirroring the board, so a
    transformed position keeps the same outcome. 'turno' is left unchanged.
    """
    tableros = atributos[:, :64].reshape(-1, 8, 8)
    turnos = atributos[:, 64:]
    variantes = []
    for espejo in (False, True):
        base = tableros[:, :, ::-1] if espejo else tableros
        for giros in range(4):
            girados = np.rot90(base, k=giros, axes=(1, 2)).reshape(-1, 64)
            variantes.append(np.hstack([girados, turnos]))
    return np.vstack(variantes), np.tile(objetivo, 8)


atributos_entrenamiento = atributos.iloc[indices_entrenamiento].to_numpy(dtype=float)
atributos_validacion = atributos.iloc[indices_validacion].to_numpy(dtype=float)
atributos_prueba = atributos.iloc[indices_prueba].to_numpy(dtype=float)
objetivo_entrenamiento = objetivo[indices_entrenamiento]
objetivo_validacion = objetivo[indices_validacion]
objetivo_prueba = objetivo[indices_prueba]

# Augment only the training set: validation and test stay as real positions.
atributos_entrenamiento, objetivo_entrenamiento = aumentar_con_simetrias(
    atributos_entrenamiento, objetivo_entrenamiento
)
print(f"entrenamiento con simetrias: {len(objetivo_entrenamiento)} posiciones")

# REFERENCIAS SIN APRENDIZAJE
# For MSE, the best constant is the training mean; for picking the winner,
# compare against always naming the side that won most training games.
media_entrenamiento = float(np.mean(objetivo_entrenamiento))
mse_constante_cero = float(np.mean(objetivo_prueba ** 2))
mse_media = float(np.mean((objetivo_prueba - media_entrenamiento) ** 2))
decididas = objetivo_prueba != 0
lado_mas_ganador = 1.0 if np.sum(objetivo_entrenamiento == 1) >= np.sum(objetivo_entrenamiento == -1) else -1.0
acierto_referencia = float(np.mean(objetivo_prueba[decididas] == lado_mas_ganador))

# NEURONAS ENTRADA Y SALIDA
red_otelo = Sequential()
red_otelo.add(Input(shape=(65,)))
red_otelo.add(Dense(128, activation='relu'))
red_otelo.add(Dense(64, activation='relu'))
red_otelo.add(Dense(32, activation='relu'))
red_otelo.add(Dense(1, activation='tanh'))

# 3. Compilación con SGD ajustado
optimizador = SGD(learning_rate=0.001)
red_otelo.compile(
    optimizer=optimizador,
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

# Stop when validation loss stops improving and keep the best weights seen.
parada_temprana = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

historial = red_otelo.fit(
    atributos_entrenamiento,
    objetivo_entrenamiento,
    validation_data=(atributos_validacion, objetivo_validacion),
    batch_size=32,
    epochs=2000,
    callbacks=[parada_temprana],
    verbose=1
)

mse_red, mae_red = red_otelo.evaluate(atributos_prueba, objetivo_prueba, verbose=0)
prediccion = red_otelo.predict(atributos_prueba, verbose=0).ravel()
acierto_red = float(np.mean(np.sign(prediccion[decididas]) == objetivo_prueba[decididas]))

nombre_lado = 'blancas' if lado_mas_ganador == 1 else 'negras'
print(f"\nEpocas entrenadas: {len(historial.history['loss'])}")
print(f"Siempre 0:                    MSE {mse_constante_cero:.4f}")
print(f"Siempre la media ({media_entrenamiento:+.3f}):    MSE {mse_media:.4f}")
print(f"Siempre ganan {nombre_lado}:        acierto {acierto_referencia:.3f}")
print(f"Red neuronal:                 MSE {mse_red:.4f} | MAE {mae_red:.4f} | acierto {acierto_red:.3f}")

# Winner accuracy by game phase (number of discs on the board).
fichas = (atributos_prueba[:, :64] != 0).sum(axis=1)
for minimo, maximo in [(0, 23), (24, 43), (44, 64)]:
    fase = decididas & (fichas >= minimo) & (fichas <= maximo)
    acierto_fase = float(np.mean(np.sign(prediccion[fase]) == objetivo_prueba[fase]))
    print(f"  {minimo}-{maximo} fichas: acierto {acierto_fase:.3f} ({int(fase.sum())} posiciones)")

red_otelo.save('red_otelo.h5')
