"""Genera el dataset de entrenamiento de la red con partidas MCTS contra MCTS.

Cada partida la juegan dos agentes MCTS con simulaciones aleatorias y el mismo número
de iteraciones. Se guarda el tablero después de cada jugada, el jugador que la hizo y
el ganador FINAL de la partida. La columna `partida` permite separar entrenamiento y
prueba por partidas completas (y no por posiciones sueltas) al entrenar la red.

Las partidas se reparten entre varios procesos porque el MCTS en Python puro es lento.

Uso:
    python generar_datos.py --partidas 150 --iteraciones 1000 --procesos 8
"""

import argparse
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
    parser.add_argument("--partidas", type=int, default=150)
    parser.add_argument("--iteraciones", type=int, default=1000)
    parser.add_argument("--procesos", type=int, default=8)
    parser.add_argument("--semilla", type=int, default=394867)
    parser.add_argument("--salida", default="partidas_ia_vs_ia.csv")
    args = parser.parse_args()

    tareas = [(i, args.iteraciones, args.semilla + i) for i in range(args.partidas)]
    todas_las_filas = []
    resultados = {0: 0, 1: 0, 2: 0}
    inicio = time.time()

    with Pool(args.procesos) as pool:
        for hechas, (filas, ganador, negras, blancas) in enumerate(pool.imap_unordered(jugar_partida, tareas), start=1):
            todas_las_filas.extend(filas)
            resultados[ganador] += 1
            minutos = (time.time() - inicio) / 60
            restante = minutos / hechas * (args.partidas - hechas)
            print(f"[{hechas}/{args.partidas}] negras {negras} - blancas {blancas} | "
                  f"{minutos:.1f} min, quedan ~{restante:.0f} min", flush=True)

    columnas = [str((i, j)) for i in range(8) for j in range(8)] + ["turno", "ganador", "partida"]
    df = pd.DataFrame(todas_las_filas, columns=columnas).sort_values(["partida"], kind="stable")
    df.to_csv(args.salida, index=False)

    print(f"\n{len(df)} posiciones de {args.partidas} partidas guardadas en {args.salida}")
    print(f"Ganan negras: {resultados[1]}, ganan blancas: {resultados[2]}, empates: {resultados[0]}")


if __name__ == "__main__":
    main()
