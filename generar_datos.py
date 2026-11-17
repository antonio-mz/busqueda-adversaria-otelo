"""Genera el dataset de entrenamiento de la red con partidas MCTS contra MCTS.

Cada partida la juegan dos agentes MCTS con simulaciones aleatorias y el mismo número
de iteraciones. Se guarda el tablero después de cada jugada, el jugador que la hizo y
el ganador FINAL de la partida. La columna `partida` permite separar entrenamiento y
prueba por partidas completas (y no por posiciones sueltas) al entrenar la red.

Las partidas se reparten entre varios procesos porque el MCTS en Python puro es lento.

Each game is appended to the output file as soon as it finishes, so an interrupted run
keeps its finished games; running the same command again skips the games already saved.
Game numbers go from --inicio to --inicio + --partidas - 1 and each game uses the seed
--semilla + its number, so a new batch must start after the last saved game number.

Uso:
    python generar_datos.py --partidas 150 --iteraciones 1000 --procesos 8
    python generar_datos.py --inicio 150 --partidas 600 --salida partidas_extra.csv
"""

import argparse
import os
import random
import time
from multiprocessing import Pool

import pandas as pd

import mcts
import otelo


def jugar_partida(args):
    """Juega una partida MCTS contra MCTS y devuelve sus filas etiquetadas."""
    num_partida, iteraciones, semilla = args
    random.seed(semilla)

    tablero = otelo.crear_tablero()
    turno = 1
    estados = []

    while True:
        if not otelo.jugador_puede_mover(tablero, 1) and not otelo.jugador_puede_mover(tablero, 2):
            ganador, blancas, negras = otelo.ganador(tablero)
            break

        if not otelo.jugador_puede_mover(tablero, turno):
            turno = 3 - turno
            continue

        movimiento = mcts.mcts_uct(tablero, turno, iteraciones=iteraciones)
        otelo.poner_ficha(tablero, movimiento[0], movimiento[1], turno)
        estados.append((tablero.copy(), turno))
        turno = 3 - turno

    filas = []
    for estado, jugador in estados:
        fila = {str(casilla): valor for casilla, valor in estado.items()}
        fila["turno"] = jugador
        fila["ganador"] = ganador
        fila["partida"] = num_partida
        filas.append(fila)
    return filas, ganador, negras, blancas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inicio", type=int, default=0, help="number of the first game")
    parser.add_argument("--partidas", type=int, default=150)
    parser.add_argument("--iteraciones", type=int, default=1000)
    parser.add_argument("--procesos", type=int, default=8)
    parser.add_argument("--semilla", type=int, default=394867)
    parser.add_argument("--salida", default="partidas_ia_vs_ia.csv")
    args = parser.parse_args()

    columnas = [str((i, j)) for i in range(8) for j in range(8)] + ["turno", "ganador", "partida"]

    # Resume: skip the games this output file already holds.
    ya_guardadas = set()
    if os.path.exists(args.salida):
        existente = pd.read_csv(args.salida)
        if list(existente.columns) != columnas:
            raise SystemExit(f"{args.salida} exists with different columns; choose another --salida")
        ya_guardadas = set(existente["partida"].unique())

    numeros = range(args.inicio, args.inicio + args.partidas)
    tareas = [(n, args.iteraciones, args.semilla + n) for n in numeros if n not in ya_guardadas]
    if ya_guardadas:
        print(f"{len(ya_guardadas)} partidas ya guardadas en {args.salida}; quedan {len(tareas)}")

    resultados = {0: 0, 1: 0, 2: 0}
    posiciones = 0
    inicio = time.time()

    with Pool(args.procesos) as pool:
        for hechas, (filas, ganador, negras, blancas) in enumerate(pool.imap_unordered(jugar_partida, tareas), start=1):
            df = pd.DataFrame(filas, columns=columnas)
            escribir_cabecera = not os.path.exists(args.salida)
            df.to_csv(args.salida, mode="a", header=escribir_cabecera, index=False)
            posiciones += len(df)
            resultados[ganador] += 1
            minutos = (time.time() - inicio) / 60
            restante = minutos / hechas * (len(tareas) - hechas)
            print(f"[{hechas}/{len(tareas)}] negras {negras} - blancas {blancas} | "
                  f"{minutos:.1f} min, quedan ~{restante:.0f} min", flush=True)

    print(f"\n{posiciones} posiciones de {len(tareas)} partidas nuevas guardadas en {args.salida}")
    print(f"Ganan negras: {resultados[1]}, ganan blancas: {resultados[2]}, empates: {resultados[0]}")


if __name__ == "__main__":
    main()
